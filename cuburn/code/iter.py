"""
The main iteration loop.
"""

from cuburn.code import mwc, variations
from cuburn.code.util import *

class IterCode(HunkOCode):
    # The number of threads per block
    NTHREADS = 512

    def __init__(self, features):
        self.features = features
        self.packer = DataPacker('iter_info')
        iterbody = self._iterbody()
        bodies = [self._xfbody(i,x) for i,x in enumerate(self.features.xforms)]
        bodies.append(iterbody)
        self.defs = '\n'.join(bodies)

    decls = """
// Note: for normalized lookups, uchar4 actually returns floats
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> palTex;
__shared__ iter_info info;
"""

    def _xfbody(self, xfid, xform):
        px = self.packer.view('info', 'xf%d_' % xfid)
        px.sub('xf', 'cp.xforms[%d]' % xfid)

        tmpl = Template("""
__device__
void apply_xf{{xfid}}(float *ix, float *iy, float *icolor, mwc_st *rctx) {
    float tx, ty, ox = *ix, oy = *iy;

    {{apply_affine_flam3('ox', 'oy', 'tx', 'ty', px, 'xf.c', 'pre')}}

    ox = 0;
    oy = 0;

    {{for v in xform.vars}}
    {{if variations.var_nos[v].startswith('pre_')}}
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]].substitute(locals())}}
    }
    {{endif}}
    {{endfor}}

    {{for v in xform.vars}}
    {{if not variations.var_nos[v].startswith('pre_')}}
    if (1) {
        float w = {{px.get('xf.var[%d]' % v)}};
        {{variations.var_code[variations.var_nos[v]].substitute(locals())}}
    }
    {{endif}}
    {{endfor}}

    *ix = ox;
    *iy = oy;

    float csp = {{px.get('xf.color_speed')}};
    *icolor = *icolor * (1.0f - csp) + {{px.get('xf.color')}} * csp;
};
""")
        g = dict(globals())
        g.update(locals())
        return tmpl.substitute(g)

    def _iterbody(self):
        tmpl = Template('''
__global__
void iter(mwc_st *msts, iter_info *infos, float4 *accbuf, float *denbuf) {
    mwc_st rctx = msts[gtid()];
    iter_info *info_glob = &(infos[blockIdx.x]);

    // load info to shared memory cooperatively
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i * 4 < sizeof(iter_info); i += blockDim.x * blockDim.y)
        reinterpret_cast<float*>(&info)[i] =
            reinterpret_cast<float*>(info_glob)[i];

    int consec_bad = -{{features.fuse}};
    // TODO: remove '512' constant
    int nsamps = {{packer.get("cp.width * cp.height / (cp.ntemporal_samples * 512.) * cp.adj_density")}};

    float x, y, color;
    int last_xf_used = 0;
    x = mwc_next_11(&rctx);
    y = mwc_next_11(&rctx);
    color = mwc_next_01(&rctx);

    while (nsamps > 0) {
        float xfsel = mwc_next_01(&rctx);

        {{if packer.get("cp.chaos_used")==True}}
        {{for density_row_idx, prior_xform_idx in enumerate(features.std_xforms)}}
        {{for density_col_idx,  this_xform_idx in enumerate(features.std_xforms)}}
        if (last_xf_used == {{prior_xform_idx}} && 
                xfsel < {{packer.get("cp.chaos_densities[%d][%d]" % (density_row_idx, density_col_idx))}}) {
            apply_xf{{this_xform_idx}}(&x, &y, &color, &rctx);
            last_xf_used = {{this_xform_idx}};
        } else
        {{endfor}}
        {{endfor}}
        {{else}}
        {{for density_col_idx,  this_xform_idx in enumerate(features.std_xforms)}}
        if (xfsel < {{packer.get("cp.norm_density[%d]" % (density_col_idx))}}) {
            apply_xf{{this_xform_idx}}(&x, &y, &color, &rctx);
        } else
        {{endfor}}
        {{endif}}
        {
            //printf("%d ",last_xf_used);
            denbuf[0] = xfsel;
            break; // TODO: fail here
        }

        if (consec_bad < 0) {
            consec_bad++;
            continue;
        }
        nsamps--;

        {{if features.final_xform_index}}
        float fx = x, fy = y, fcolor;
        apply_xf{{features.final_xform_index}}(&fx, &fy, &fcolor, &rctx);
        {{endif}}

        // TODO: this may not optimize well, verify.
        float cx, cy;
        {{if features.final_xform_index}}
        {{apply_affine('fx', 'fy', 'cx', 'cy', packer,
                       'cp.camera_transform', 'cam')}}
        {{else}}
        {{apply_affine('x', 'y', 'cx', 'cy', packer,
                       'cp.camera_transform', 'cam')}}
        {{endif}}

        // TODO: verify that constants get premultiplied
        float ditherwidth = {{packer.get("0.33 * cp.spatial_filter_radius")}};
        float u0 = mwc_next_01(&rctx);
        float r = ditherwidth * sqrt(-2.0f * log2f(u0) / M_LOG2E);

        // TODO: provide mwc_next_0_2pi()
        float u1 = 2.0f * M_PI * mwc_next_01(&rctx);

        float ditherx = r * cos(u1);
        float dithery = r * sin(u1);
        int ix = trunca(cx+ditherx), iy = trunca(cy+dithery);

        if (ix < 0 || ix >= {{features.acc_width}} ||
            iy < 0 || iy >= {{features.acc_height}} ) {
            consec_bad++;
            if (consec_bad > {{features.max_oob}}) {
                x = mwc_next_11(&rctx);
                y = mwc_next_11(&rctx);
                color = mwc_next_01(&rctx);
                consec_bad = -{{features.fuse}};
            }
            continue;
        }

        int i = iy * {{features.acc_stride}} + ix;

        float4 outcol = tex2D(palTex, color, {{packer.get("cp_step_frac")}});
        float4 pix = accbuf[i];
        pix.x += outcol.x;
        pix.y += outcol.y;
        pix.z += outcol.z;
        pix.w += outcol.w;
        accbuf[i] = pix;    // TODO: atomic operations (or better)
        denbuf[i] += 1.0f;
    }
    asm volatile ("membar.cta;");
}
''')
        return tmpl.substitute(
                features = self.features,
                packer = self.packer.view('info'),
                **globals())

