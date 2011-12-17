import numpy as np

from util import Template

var_code = {}
var_params = {}

def var(num, name, code, params=None):
    var_code[name] = Template(code, name)
    if params is not None:
        r = {}
        for p in params.split():
            if '=' in p:
                p, default = p.split('=')
                if default == 'M_PI':
                    default = np.pi
                else:
                    default = float(default)
            else:
                default = 0.0
            r[p] = default
        var_params[name] = r

# TODO: This is a shitty hack
def precalc(name, code):
    def precalc_fun(pv):
        pre = pv._precalc()
        pre._code(Template(code, name+'_precalc').substitute(pre=pre))
    Template.default_namespace[name+'_precalc'] = precalc_fun

# Variables note: all functions will have their weights as 'w',
# input variables 'tx' and 'ty', and output 'ox' and 'oy' available
# from the calling context. Each statement will be placed inside brackets,
# to avoid namespace pollution.
var(0, 'linear', """
    ox += tx * w;
    oy += ty * w;
    """)

var(1, 'sinusoidal', """
    ox += w * sinf(tx);
    oy += w * sinf(ty);
    """)

var(2, 'spherical', """
    float r2 = w / (tx*tx + ty*ty);
    ox += tx * r2;
    oy += ty * r2;
    """)

var(3, 'swirl', """
    float r2 = tx*tx + ty*ty;
    float c1 = sinf(r2);
    float c2 = cosf(r2);
    ox += w * (c1*tx - c2*ty);
    oy += w * (c2*tx + c1*ty);
    """)

var(4, 'horseshoe', """
    float r = w / sqrtf(tx*tx + ty*ty);
    ox += r * (tx - ty) * (tx + ty);
    oy += 2.0f * tx * ty * r;
    """)

var(5, 'polar', """
    ox += w * atan2f(tx, ty) * M_1_PI;
    oy += w * (sqrtf(tx * tx + ty * ty) - 1.0f);
    """)

var(6, 'handkerchief', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    ox += w * r * sinf(a+r);
    oy += w * r * cosf(a-r);
    """)

var(7, 'heart', """
    float sq = sqrtf(tx*tx + ty*ty);
    float a = sq * atan2f(tx, ty);
    float r = w * sq;
    ox += r * sinf(a);
    oy -= r * cosf(a);
    """)

var(8, 'disc', """
    float a = w * atan2f(tx, ty) * M_1_PI;
    float r = M_PI * sqrtf(tx*tx + ty*ty);
    ox += sinf(r) * a;
    oy += cosf(r) * a;
    """)

var(9, 'spiral', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    float r1 = w / r;
    ox += r1 * (cosf(a) + sinf(r));
    oy += r1 * (sinf(a) - cosf(r));
    """)

var(10, 'hyperbolic', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    ox += w * sinf(a) / r;
    oy += w * cosf(a) * r;
    """)

var(11, 'diamond', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    ox += w * sinf(a) * cosf(r);
    oy += w * cosf(a) * sinf(r);
    """)

var(12, 'ex', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    float n0 = sinf(a+r);
    float n1 = cosf(a-r);
    float m0 = n0*n0*n0*r;
    float m1 = n1*n1*n1*r;
    ox += w * (m0 + m1);
    oy += w * (m0 - m1);
    """)

var(13, 'julia', """
    float a = 0.5f * atan2f(tx, ty);
    if (mwc_next(rctx) & 1) a += M_PI;
    float r = w * sqrtf(sqrtf(tx*tx + ty*ty)); // TODO: fastest?
    ox += r * cosf(a);
    oy += r * sinf(a);
    """)

var(14, 'bent', """
    float nx = 1.0f;
    if (tx < 0.0f) nx = 2.0f;
    float ny = 1.0f;
    if (ty < 0.0f) ny = 0.5f;
    ox += w * nx * tx;
    oy += w * ny * ty;
    """)

var(15, 'waves', """
    float c10 = {{px.affine.xy}};
    float c11 = {{px.affine.yy}};
    float dx = {{px.affine.xo}};
    float dy = {{px.affine.yo}};
    float dx2 = 1.0f / (dx * dx);
    float dy2 = 1.0f / (dy * dy);

    ox += w * (tx + c10 * sinf(ty * dx2));
    oy += w * (ty + c11 * sinf(tx * dy2));
    """)

var(16, 'fisheye', """
    float r = sqrtf(tx*tx + ty*ty);
    r = 2.0f * w / (r + 1.0f);
    ox += r * ty;
    oy += r * tx;
    """)

var(17, 'popcorn', """
    float dx = tanf(3.0f*ty);
    float dy = tanf(3.0f*tx);
    ox += w * (tx + {{px.affine.xo}} * sinf(dx));
    oy += w * (ty + {{px.affine.yo}} * sinf(dy));
    """)

var(18, 'exponential', """
    float dx = w * expf(tx - 1.0f);
    float dy = M_PI * ty;
    ox += dx * cosf(dy);
    oy += dx * sinf(dy);
    """)

var(19, 'power', """
    float a = atan2f(tx, ty);
    float sa = sinf(a);
    float r = w * powf(sqrtf(tx*tx + ty*ty),sa);
    ox += r * cosf(a);
    oy += r * sa;
    """)

var(20, 'cosine', """
    float a = M_PI * tx;
    ox += w * cosf(a) * coshf(ty);
    oy -= w * sinf(a) * sinhf(ty);
    """)

var(21, 'rings', """
    float dx = {{px.affine.xo}} * {{px.affine.xo}};
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    r = w * (fmodf(r+dx, 2.0f*dx) - dx + r * (1.0f - dx));
    ox += r * cosf(a);
    oy += r * sinf(a);
    """)

var(22, 'fan', """
    float dx = M_PI * ({{px.affine.xo}} * {{px.affine.xo}});
    float dx2 = 0.5f * dx;
    float dy = {{px.affine.yo}};
    float a = atan2f(tx, ty);
    a += (fmodf(a+dy, dx) > dx2) ? -dx2 : dx2;
    float r = w * sqrtf(tx*tx + ty*ty);
    ox += r * cosf(a);
    oy += r * sinf(a);
    """)

var(23, 'blob', """
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    float bdiff = 0.5f * ({{pv.high}} - {{pv.low}});
    r *= w * ({{pv.low}} + bdiff * (1.0f + sinf({{pv.waves}} * a)));
    ox += sinf(a) * r;
    oy += cosf(a) * r;
    """, 'low high=1 waves=1')

var(24, 'pdj', """
    float nx1 = cosf({{pv.b}} * tx);
    float nx2 = sinf({{pv.c}} * tx);
    float ny1 = sinf({{pv.a}} * ty);
    float ny2 = cosf({{pv.d}} * ty);
    ox += w * (ny1 - nx1);
    oy += w * (nx2 - ny2);
    """, 'a b c d')

var(25, 'fan2', """
    float dy = {{pv.y}};
    float dx = M_PI * {{pv.x}} * {{pv.x}};
    float dx2 = 0.5f * dx;
    float a = atan2f(tx, ty);
    float r = w * sqrtf(tx*tx + ty*ty);
    float t = a + dy - dx * truncf((a + dy)/dx);
    if (t > dx2)
        a -= dx2;
    else
        a += dx2;

    ox += r * sinf(a);
    oy += r * cosf(a);
    """, 'x y')

var(26, 'rings2', """
    float dx = {{pv.val}} * {{pv.val}};
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    r += -2.0f * dx * (int)((r+dx)/(2.0f*dx)) + r * (1.0f - dx);
    ox += w * sinf(a) * r;
    oy += w * cosf(a) * r;
    """, 'val')

var(27, 'eyefish', """
    float r = 2.0f * w / (sqrtf(tx*tx + ty*ty) + 1.0f);
    ox += r * tx;
    oy += r * ty;
    """)

var(28, 'bubble', """
    float r = w / (0.25f * (tx*tx + ty*ty) + 1.0f);
    ox += r * tx;
    oy += r * ty;
    """)

var(29, 'cylinder', """
    ox += w * sinf(tx);
    oy += w * ty;
    """)

precalc('perspective', """
        float pang = {{pre.angle}} * M_PI_2;
        float pdist = fmaxf(1e-9, {{pre.dist}});
        {{pre._set('mdist')}} = pdist;
        {{pre._set('sin')}} = sin(pang);
        {{pre._set('cos')}} = pdist * cos(pang);
        """)

var(30, 'perspective', """
    {{perspective_precalc(pv)}}

    float t = 1.0f / ({{pv.mdist}} - ty * {{pv.sin}});
    ox += w * {{pv.mdist}} * tx * t;
    oy += w * {{pv.cos}} * ty * t;
    """, 'angle dist')

var(31, 'noise', """
    float tmpr = mwc_next_01(rctx) * 2.0f * M_PI;
    float r = w * mwc_next_01(rctx);
    ox += tx * r * cosf(tmpr);
    oy += ty * r * sinf(tmpr);
    """)

precalc('julian',
        "{{pre._set('cn')}} = {{pre.dist}} / (2.0f * {{pre.power}});\n")

var(32, 'julian', """
    {{julian_precalc(pv)}}
    float power = {{pv.power}};
    float t_rnd = truncf(mwc_next_01(rctx) * fabsf(power));
    float a = atan2f(ty, tx);
    float tmpr = (a + 2.0f * M_PI * t_rnd) / power;
    float cn = {{pv.cn}};
    float r = w * powf(tx * tx + ty * ty, cn);

    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
    """, 'power=1 dist=1')

precalc('juliascope',
        "{{pre._set('cn')}} = {{pre.dist}} / (2.0f * {{pre.power}});\n")

var(33, 'juliascope', """
    {{juliascope_precalc(pv)}}

    float ang = atan2f(ty, tx);
    float power = {{pv.power}};
    float t_rnd = truncf(mwc_next_01(rctx) * fabsf(power));
    // TODO: don't draw the extra random number
    if (mwc_next(rctx) & 1) ang = -ang;
    float tmpr = (2.0f * M_PI * t_rnd + ang) / power;
    float r = w * powf(tx * tx + ty * ty, {{pv.cn}});

    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
    """, 'power=1 dist=1')

var(34, 'blur', """
    float tmpr = mwc_next_01(rctx) * 2.0f * M_PI;
    float r = w * mwc_next_01(rctx);
    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
    """)

var(35, 'gaussian_blur', """
    float ang = mwc_next_01(rctx) * 2.0f * M_PI;
    // constant factor here is stdev correction for converting to Box-Muller;
    // np.std(np.sum(np.random.random((1<<30, 4)), axis=1) - 2)
    // TODO: maybe derive it analytically
    float r = w * 0.57736 * sqrtf(-2.0f * log2f(mwc_next_01(rctx)) / M_LOG2E);
    ox += r * cosf(ang);
    oy += r * sinf(ang);
    """)

var(36, 'radial_blur', """
    float blur_angle = {{pv.angle}} * M_PI * 0.5f;
    float spinvar = sinf(blur_angle);
    float zoomvar = cosf(blur_angle);

    float r = w * 0.57736 * sqrtf(-2.0f * log2f(mwc_next_01(rctx)) / M_LOG2E);
    float ra = sqrtf(tx*tx + ty*ty);
    float tmpa = atan2f(ty, tx) + spinvar * r;
    float rz = zoomvar * r - 1.0f;
    ox += ra*cosf(tmpa) + rz*tx;
    oy += ra*sinf(tmpa) + rz*ty;
    """, 'angle')

var(37, 'pie', """
    float slices = {{pv.slices}};
    float sl = truncf(mwc_next_01(rctx) * slices + 0.5f);
    float a = {{pv.rotation}} +
                2.0f * M_PI * (sl + mwc_next_01(rctx) * {{pv.thickness}}) / slices;
    float r = w * mwc_next_01(rctx);
    ox += r * cosf(a);
    oy += r * sinf(a);
    """, 'slices=6 rotation thickness=0.5')

var(38, 'ngon', """
    float power = {{pv.power}} * 0.5f;
    float b = 2.0f * M_PI / {{pv.sides}};
    float corners = {{pv.corners}};
    float circle = {{pv.circle}};

    float r_factor = powf(tx*tx + ty*ty, power);
    float theta = atan2f(ty, tx);
    float phi = theta - b * floorf(theta/b);
    if (phi > b/2.0f) phi -= b;
    float amp = (corners * (1.0f / cosf(phi) - 1.0f) + circle) / r_factor;

    ox += w * tx * amp;
    oy += w * ty * amp;
    """, 'sides=5 power=3 circle=1 corners=2')

var(39, 'curl', """
    float c1 = {{pv.c1}};
    float c2 = {{pv.c2}};

    float re = 1.0f + c1*tx + c2*(tx*tx - ty*ty);
    float im = c1*ty + 2.0f*c2*tx*ty;
    float r = w / (re*re + im*im);

    ox += r * (tx*re + ty*im);
    oy += r * (ty*re - tx*im);
    """, 'c1=1 c2')

var(40, 'rectangles', """
    float rx = {{pv.x}};
    float ry = {{pv.y}};

    ox += w * ( (rx==0.0f) ? tx : rx * (2.0f * floorf(tx/rx) + 1.0f) - tx);
    oy += w * ( (ry==0.0f) ? ty : ry * (2.0f * floorf(ty/ry) + 1.0f) - ty);
    """, 'x y')

var(41, 'arch', """
    float ang = mwc_next_01(rctx) * w * M_PI;

    ox += w * sinf(ang);
    oy += w * sinf(ang) * sinf(ang) / cosf(ang);
    """)

var(42, 'tangent', """
    ox += w * sinf(tx) / cosf(ty);
    oy += w * tanf(ty);
    """)

var(43, 'square', """
    ox += w * (mwc_next_01(rctx) - 0.5f);
    oy += w * (mwc_next_01(rctx) - 0.5f);
    """)

var(44, 'rays', """
    float ang = w * mwc_next_01(rctx) * M_PI;
    float r = w / (tx*tx + ty*ty);
    float tanr = w * tanf(ang) * r;
    ox += tanr * cosf(tx);
    oy += tanr * sinf(ty);
    """)

var(45, 'blade', """
    float r = mwc_next_01(rctx) * w * sqrtf(tx*tx + ty*ty);
    ox += w * tx * (cosf(r) + sinf(r));
    oy += w * tx * (cosf(r) - sinf(r));
    """)

var(46, 'secant2', """
    float r = w * sqrtf(tx*tx + ty*ty);
    float cr = cosf(r);
    float icr = 1.0f / cr;
    icr += (cr < 0 ? 1 : -1);

    ox += w * tx;
    oy += w * icr;
    """)

# var 47 is twintrian, has a call to badvalue in it

var(48, 'cross', """
    float s = tx*tx - ty*ty;
    float r = w * sqrtf(1.0f / (s*s));

    ox += r * tx;
    oy += r * ty;
    """)

var(49, 'disc2', """
    float twist = {{pv.twist}};
    float rotpi = {{pv.rot}} * M_PI;

    float sintwist = sinf(twist);
    float costwist = cosf(twist) - 1.0f;

    if (twist > 2.0f * M_PI) {
        float k = (1.0f + twist - 2.0f * M_PI);
        sintwist *= k;
        costwist *= k;
    }

    if (twist < -2.0f * M_PI) {
        float k = (1.0f + twist + 2.0f * M_PI);
        sintwist *= k;
        costwist *= k;
    }

    float t = rotpi * (tx + ty);
    float r = w * atan2f(tx, ty) / M_PI;

    ox += r * (sinf(t) + costwist);
    oy += r * (cosf(t) + sintwist);
    """, 'rot twist')

var(50, 'super_shape', """
    float ang = atan2f(ty, tx);
    float theta = 0.25f * ({{pv.m}} * ang + M_PI);
    float t1 = fabsf(cosf(theta));
    float t2 = fabsf(sinf(theta));
    t1 = powf(t1, {{pv.n2}});
    t2 = powf(t2, {{pv.n3}});
    float myrnd = {{pv.rnd}};
    float d = sqrtf(tx*tx+ty*ty);

    float r = w * ((myrnd*mwc_next_01(rctx) + (1.0f-myrnd)*d) - {{pv.holes}})
                * powf(t1+t2, -1.0f / {{pv.n1}}) / d;

    ox += r * tx;
    oy += r * ty;
    """, 'rnd m n1=1 n2=1 n3=1 holes')

var(51, 'flower', """
    float holes = {{pv.holes}};
    float petals = {{pv.petals}};

    float r = w * (mwc_next_01(rctx) - holes)
                * cosf(petals*atan2f(ty, tx)) / sqrtf(tx*tx + ty*ty);

    ox += r * tx;
    oy += r * ty;
    """, 'holes petals')

var(52, 'conic', """
    float d = sqrtf(tx*tx + ty*ty);
    float ct = tx / d;
    float holes = {{pv.holes}};
    float eccen = {{pv.eccentricity}};

    float r = w * (mwc_next_01(rctx) - holes) * eccen / (1.0f + eccen*ct) / d;

    ox += r * tx;
    oy += r * ty;
    """, 'holes eccentricity=1')

var(53, 'parabola', """
    float r = sqrtf(tx*tx + ty*ty);
    float sr = sinf(r);
    float cr = cosf(r);

    ox += {{pv.height}} * w * sr * sr * mwc_next_01(rctx);
    oy += {{pv.width}}  * w * cr * mwc_next_01(rctx);
    """, 'height width')

var(54, 'bent2', """
    float nx = 1.0f;
    if (tx < 0.0f) nx = {{pv.x}};
    float ny = 1.0f;
    if (ty < 0.0f) ny = {{pv.y}};
    ox += w * nx * tx;
    oy += w * ny * ty;
    """, 'x=1 y=1')

var(55, 'bipolar', """
    float x2y2 = tx*tx + ty*ty;
    float t = x2y2 + 1.0f;
    float x2 = tx * 2.0f;
    float ps = -M_PI_2 * {{pv.shift}};
    float y = 0.5f * atan2f(2.0f * ty, x2y2 - 1.0f) + ps;

    if (y > M_PI_2)
        y = -M_PI_2 + fmodf(y + M_PI_2, M_PI);
    else if (y < -M_PI_2)
        y = M_PI_2 - fmodf(M_PI_2 - y, M_PI);

    ox += w * 0.25f * M_2_PI * logf( (t+x2) / (t-x2) );
    oy += w * M_2_PI * y;
    """, 'shift')

var(56, 'boarders', """
    float roundX = rintf(tx);
    float roundY = rintf(ty);
    float offsetX = tx - roundX;
    float offsetY = ty - roundY;

    if (mwc_next_01(rctx) > 0.75f) {
        ox += w * (offsetX*0.5f + roundX);
        oy += w * (offsetY*0.5f + roundY);
    } else {
        if (fabsf(offsetX) >= fabsf(offsetY)) {
            if (offsetX >= 0.0f) {
                ox += w * (offsetX*0.5f + roundX + 0.25f);
                oy += w * (offsetY*0.5f + roundY + 0.25f * offsetY / offsetX);
            } else {
                ox += w * (offsetX*0.5f + roundX - 0.25f);
                oy += w * (offsetY*0.5f + roundY - 0.25f * offsetY / offsetX);
            }
        } else {
            if (offsetY >= 0.0f) {
                oy += w * (offsetY*0.5f + roundY + 0.25f);
                ox += w * (offsetX*0.5f + roundX + offsetX / offsetY * 0.25f);
            } else {
                oy += w * (offsetY*0.5f + roundY - 0.25f);
                ox += w * (offsetX*0.5f + roundX - offsetX / offsetY * 0.25f);
            }
        }
    }
    """)

var(57, 'butterfly', """
    /* wx is weight*4/sqrt(3*pi) */
    float wx = w * 1.3029400317411197908970256609023f;
    float y2 = ty * 2.0f;
    float r = wx * sqrtf(fabsf(ty * tx)/(tx*tx + y2*y2));
    ox += r * tx;
    oy += r * y2;
    """)

var(58, 'cell', """
    float cell_size = {{pv.size}};
    float inv_cell_size = 1.0f/cell_size;

    /* calculate input cell */
    float x = floorf(tx * inv_cell_size);
    float y = floorf(ty * inv_cell_size);

    /* Offset from cell origin */
    float dx = tx - x*cell_size;
    float dy = ty - y*cell_size;

   /* interleave cells */
    if (y >= 0.0f) {
        if (x >= 0.0f) {
            y *= 2.0f;
            x *= 2.0f;
        } else {
            y *= 2.0f;
            x = -(2.0f*x+1.0f);
        }
    } else {
        if (x >= 0.0f) {
            y = -(2.0f*y+1.0f);
            x *= 2.0f;
         } else {
            y = -(2.0f*y+1.0f);
            x = -(2.0f*x+1.0f);
         }
    }

    ox += w * (dx + x*cell_size);
    oy -= w * (dy + y*cell_size);
    """, 'size=1')

var(59, 'cpow', """
    float a = atan2f(ty, tx);
    float lnr = 0.5f * logf(tx*tx+ty*ty);
    float power = {{pv.power}};
    float va = 2.0f * M_PI / power;
    float vc = {{pv.cpow_r}} / power;
    float vd = {{pv.cpow_i}} / power;
    float ang = vc*a + vd*lnr + va*floorf(power*mwc_next_01(rctx));
    float m = w * expf(vc * lnr - vd * a);
    ox += m * cosf(ang);
    oy += m * sinf(ang);
    """, 'r=1 i power=1')


precalc('curve', '''
        float xl = {{pv.xlength}}, yl = {{pv.ylength}};
        {{pre._set('x2')}} = 1.0f / max(1e-20f, xl * xl);
        {{pre._set('y2')}} = 1.0f / max(1e-20f, yl * yl);
        ''')

var(60, 'curve', """
    {{curve_precalc()}}
    float pc_xlen = {{pv.x2}}, pc_ylen = {{pv.y2}};

    ox += w * (tx + {{pv.xamp}} * expf(-ty*ty*pc_xlen));
    oy += w * (ty + {{pv.yamp}} * expf(-tx*tx*pc_ylen));
    """, 'xamp yamp xlength=1 ylength=1')

var(61, 'edisc', """
    float tmp = tx*tx + ty*ty + 1.0f;
    float tmp2 = 2.0f * tx;
    float r1 = sqrtf(tmp+tmp2);
    float r2 = sqrtf(tmp-tmp2);
    float xmax = (r1+r2) * 0.5f;
    float a1 = logf(xmax + sqrtf(xmax - 1.0f));
    float a2 = -acosf(tx/xmax);
    float neww = w / 11.57034632f;

    float snv = sinf(a1);
    float csv = cosf(a1);
    if (ty > 0.0f) snv = -snv;

    ox += neww * coshf(a2) * csv;
    oy += neww * sinhf(a2) * snv;
    """)

var(62, 'elliptic', """
    float tmp = tx*tx + ty*ty + 1.0f;
    float x2 = 2.0f * tx;
    float xmax = 0.5f * (sqrtf(tmp+x2) + sqrtf(tmp-x2));
    float a = tx / xmax;
    float b = 1.0f - a*a;
    float ssx = xmax - 1.0f;
    float neww = w / M_PI_2;

    if (b < 0.0f)
        b = 0.0f;
    else
        b = sqrtf(b);

    if (ssx < 0.0f)
        ssx = 0.0f;
    else
        ssx = sqrtf(ssx);

    ox += neww * atan2f(a,b);

    if (ty > 0.0f)
        oy += neww * logf(xmax + ssx);
    else
        oy -= neww * logf(xmax + ssx);
    """)

var(63, 'escher', """
    float a = atan2f(ty,tx);
    float lnr = 0.5f * logf(tx*tx + ty*ty);
    float ebeta = {{pv.beta}};
    float seb = sinf(ebeta);
    float ceb = cosf(ebeta);
    float vc = 0.5f * (1.0f + ceb);
    float vd = 0.5f * seb;
    float m = w * expf(vc*lnr - vd*a);
    float n = vc*a + vd*lnr;

    ox += m * cosf(n);
    oy += m * sinf(n);
    """, 'beta')

var(64, 'foci', """
    float expx = expf(tx) * 0.5f;
    float expnx = 0.25f / expx;
    float sn = sinf(ty);
    float cn = cosf(ty);
    float tmp = w / (expx + expnx - cn);
    ox += tmp * (expx - expnx);
    oy += tmp * sn;
    """)

var(65, 'lazysusan', """
    float lx = {{pv.x}};
    float ly = {{pv.y}};
    float x = tx - lx;
    float y = ty + ly;
    float r = sqrtf(x*x + y*y);

    if (r < w) {
        float a = atan2f(y,x) + {{pv.spin}} + {{pv.twist}} * (w - r);

        ox += w * (r * cosf(a) + lx);
        oy += w * (r * sinf(a) - ly);

    } else {
        r = (1.0f + {{pv.space}} / r);

        ox += w * (r * x + lx);
        oy += w * (r * y - ly);
    }
    """, 'x y twist space spin')

var(66, 'loonie', """
    float r2 = tx*tx + ty*ty;;
    float w2 = w*w;

    if (r2 < w2) {
        float r = w * sqrtf(w2/r2 - 1.0f);
        ox += r * tx;
        oy += r * ty;
    } else {
        ox += w * tx;
        oy += w * ty;
    }
    """)

var(67, 'pre_blur', """
    float rndG = w * (mwc_next_01(rctx) + mwc_next_01(rctx)
                   + mwc_next_01(rctx) + mwc_next_01(rctx) - 2.0f);
    float rndA = mwc_next_01(rctx) * 2.0f * M_PI;

    /* Note: original coordinate changed */
    tx += rndG * cosf(rndA);
    ty += rndG * sinf(rndA);
    """)

var(68, 'modulus', """
    float mx = {{pv.x}}, my = {{pv.y}};
    float xr = 2.0f*mx;
    float yr = 2.0f*my;

    if (tx > mx)
        ox += w * (-mx + fmodf(tx + mx, xr));
    else if (tx < -mx)
        ox += w * ( mx - fmodf(mx - tx, xr));
    else
        ox += w * tx;

    if (ty > my)
        oy += w * (-my + fmodf(ty + my, yr));
    else if (ty < -my)
        oy += w * ( my - fmodf(my - ty, yr));
    else
        oy += w * ty;
    """, 'x y')

var(69, 'oscope', """
    float tpf = 2.0f * M_PI * {{pv.frequency}};
    float amp = {{pv.amplitude}};
    float sep = {{pv.separation}};
    float dmp = {{pv.damping}};

    float t = amp * expf(-fabsf(tx)*dmp) * cosf(tpf*tx) + sep;

    ox += w*tx;
    if (fabsf(ty) <= t)
        oy -= w*ty;
    else
        oy += w*ty;
    """, 'separation=1 frequency=M_PI amplitude=1 damping')

var(70, 'polar2', """
    float p2v = w / M_PI;
    ox += p2v * atan2f(tx,ty);
    oy += 0.5f * p2v * logf(tx*tx + ty*ty);
    """)

var(71, 'popcorn2', """
    float c = {{pv.c}};
    ox += w * (tx + {{pv.x}} * sinf(tanf(ty*c)));
    oy += w * (ty + {{pv.y}} * sinf(tanf(tx*c)));
    """, 'x y c')

var(72, 'scry', """
    /* note that scry does not multiply by weight, but as the */
    /* values still approach 0 as the weight approaches 0, it */
    /* should be ok                                           */
    float t = tx*tx + ty*ty;
    float r = 1.0f / (sqrtf(t) * (t + 1.0f/w));
    ox += tx*r;
    oy += ty*r;
    """)

var(73, 'separation', """
    float sx2 = {{pv.x}} * {{pv.x}};
    float sy2 = {{pv.y}} * {{pv.y}};

    if (tx > 0.0f)
        ox += w * (sqrtf(tx*tx + sx2) - tx*{{pv.xinside}});
    else
        ox -= w * (sqrtf(tx*tx + sx2) + tx*{{pv.xinside}});

    if (ty > 0.0f)
        oy += w * (sqrtf(ty*ty + sy2) - ty*{{pv.yinside}});
    else
        oy -= w * (sqrtf(ty*ty + sy2) + ty*{{pv.yinside}});
    """, 'x xinside y yinside')

var(74, 'split', """
    if (cosf(tx*{{pv.xsize}}*M_PI) >= 0.0f)
        oy += w*ty;
    else
        oy -= w*ty;

    if (cosf(ty*{{pv.ysize}}*M_PI) >= 0.0f)
        ox += w*tx;
    else
        ox -= w*tx;
    """, 'xsize ysize')

var(75, 'splits', """
    ox += w*(tx + copysignf({{pv.x}}, tx));
    oy += w*(ty + copysignf({{pv.y}}, ty));
    """, 'x y')

var(76, 'stripes', """
    float roundx = floorf(tx + 0.5f);
    float offsetx = tx - roundx;
    ox += w * (offsetx * (1.0f - {{pv.space}}) + roundx);
    oy += w * (ty + offsetx*offsetx*{{pv.warp}});
    """, 'space warp')

var(77, 'wedge', """
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(ty, tx) + {{pv.swirl}} * r;
    float wc = {{pv.count}};
    float wa = {{pv.angle}};
    float c = floorf((wc * a + M_PI) * M_1_PI * 0.5f);
    float comp_fac = 1 - wa * wc * M_1_PI * 0.5f;
    a = a * comp_fac + c * wa;
    r = w * (r + {{pv.hole}});
    ox += r * cosf(a);
    oy += r * sinf(a);
    """, 'angle hole count=1 swirl')

var(80, 'whorl', """
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(ty,tx);
    
    if (r < w)
       a += {{pv.inside}} / (w - r);
    else
       a += {{pv.outside}} / (w - r);

    ox += w * r * cosf(a);
    oy += w * r * sinf(a);
    """, 'inside outside')
    
var(81, 'waves2', """
    ox += w*(tx + {{pv.scalex}}*sinf(ty * {{pv.freqx}}));
    oy += w*(ty + {{pv.scaley}}*sinf(tx * {{pv.freqy}}));
    """, 'scalex scaley freqx freqy')

var(82, 'exp', """
    float expe = expf(tx);
    ox += w * expe * cosf(ty);
    oy += w * expe * sinf(ty);
    """)

var(83, 'log', """
    ox += w * 0.5f * logf(tx*tx + ty*ty);
    oy += w * atan2f(ty, tx);
    """)

var(84, 'sin', """
    ox += w * sinf(tx) * coshf(ty);
    oy += w * cosf(tx) * sinhf(ty);
    """)

var(85, 'cos', """
    ox += w * cosf(tx) * coshf(ty);
    oy -= w * sinf(tx) * sinhf(ty);
    """)

var(86, 'tan', """
    float tanden = 1.0f/(cosf(2.0f*tx) + coshf(2.0f*ty));
    ox += w * tanden * sinf(2.0f*tx);
    oy += w * tanden * sinhf(2.0f*ty);
    """)

var(87, 'sec', """
    float secden = 2.0f/(cosf(2.0f*tx) + coshf(2.0f*ty));
    ox += w * secden * cosf(tx) * coshf(ty);
    oy += w * secden * sinf(tx) * sinhf(ty);
    """)

var(88, 'csc', """
    float cscden = 2.0f/(coshf(2.0f*ty) - cosf(2.0f*tx));
    ox += w * cscden * sinf(tx) * coshf(ty);
    oy -= w * cscden * cosf(tx) * sinhf(ty);
    """)

var(89, 'cot', """
    float cotden = 1.0f/(coshf(2.0f*ty) - cosf(2.0f*tx));
    ox += w * cotden * sinf(2.0f*tx);
    oy += w * cotden * -1.0f * sinhf(2.0f*ty);
    """)

var(90, 'sinh', """
    ox += w * sinhf(tx) * cosf(ty);
    oy += w * coshf(tx) * sinf(ty);
    """)

var(91, 'cosh', """
    ox += w * coshf(tx) * cosf(ty);
    oy += w * sinhf(tx) * sinf(ty);
    """)

var(92, 'tanh', """
    float tanhden = 1.0f/(cosf(2.0f*ty) + coshf(2.0f*tx));
    ox += w * tanhden * sinhf(2.0f*tx);
    oy += w * tanhden * sinf(2.0f*ty);
    """)

var(93, 'sech', """
    float sechden = 2.0f/(cosf(2.0f*ty) + coshf(2.0f*tx));
    ox += w * sechden * cosf(ty) * coshf(tx);
    oy -= w * sechden * sinf(ty) * sinhf(tx);
    """)

var(94, 'csch', """
    float cschden = 2.0f/(coshf(2.0f*tx) - cosf(2.0f*ty));
    ox += w * cschden * sinhf(tx) * cosf(ty);
    oy -= w * cschden * coshf(tx) * sinf(ty);
    """)

var(95, 'coth', """
    float cothden = 1.0f/(coshf(2.0f*tx) - cosf(2.0f*ty));
    ox += w * cothden * sinhf(2.0f*tx);
    oy += w * cothden * sinf(2.0f*ty);
    """)

var(97, 'flux', """
    float xpw = tx + w;
    float xmw = tx - w;
    float avgr = w * (2.0f + {{pv.spread}})
               * sqrtf(sqrtf(ty*ty+xpw*xpw)/sqrtf(ty*ty+xmw*xmw));
    float avga = (atan2f(ty, xmw) - atan2f(ty,xpw))*0.5f;
    ox += avgr * cosf(avga);
    oy += avgr * sinf(avga);
    """, 'spread')

var(98, 'mobius', """
    float rea = {{pv.re_a}};
    float ima = {{pv.im_a}};
    float reb = {{pv.re_b}};
    float imb = {{pv.im_b}};
    float rec = {{pv.re_c}};
    float imc = {{pv.im_c}};
    float red = {{pv.re_d}};
    float imd = {{pv.im_d}};

    float re_u, im_u, re_v, im_v, rad_v;

    re_u = rea * tx - ima * ty + reb;
    im_u = rea * ty + ima * tx + imb;
    re_v = rec * tx - imc * ty + red;
    im_v = rec * ty + imc * tx + imd;

    rad_v = w / (re_v*re_v + im_v*im_v);

    ox += rad_v * (re_u*re_v + im_u*im_v);
    oy += rad_v * (im_u*re_v - re_u*im_v);
    """, 're_a im_a re_b im_b re_c im_c re_d im_d')

