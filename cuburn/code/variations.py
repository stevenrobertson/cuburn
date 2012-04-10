import numpy as np

from util import Template

var_code = {}

def var(name, code, precalc=None):
    precalc_fun = None
    if precalc:
        def precalc_fun(pv, px):
            pv, px = pv._precalc(), px._precalc()
            tmpl = Template(precalc, name+'_precalc').substitute(pv=pv, px=px)
            pv._code(tmpl)
        code = "\n    {{precalc_fun(pv, px)}}" + code
    var_code[name] = Template(code, name,
                              namespace=dict(precalc_fun=precalc_fun))

# Variables note: all functions will have their weights as 'w',
# input variables 'tx' and 'ty', and output 'ox' and 'oy' available
# from the calling context. Each statement will be placed inside brackets,
# to avoid namespace pollution.
var('linear', """
    ox += tx * w;
    oy += ty * w;
""")

var('sinusoidal', """
    ox += w * sinf(tx);
    oy += w * sinf(ty);
""")

var('spherical', """
    float r2 = w / (tx*tx + ty*ty);
    ox += tx * r2;
    oy += ty * r2;
""")

var('swirl', """
    float r2 = tx*tx + ty*ty;
    float c1 = sinf(r2);
    float c2 = cosf(r2);
    ox += w * (c1*tx - c2*ty);
    oy += w * (c2*tx + c1*ty);
""")

var('horseshoe', """
    float r = w / sqrtf(tx*tx + ty*ty);
    ox += r * (tx - ty) * (tx + ty);
    oy += 2.0f * tx * ty * r;
""")

var('polar', """
    ox += w * atan2f(tx, ty) * M_1_PI;
    oy += w * (sqrtf(tx * tx + ty * ty) - 1.0f);
""")

var('handkerchief', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    ox += w * r * sinf(a+r);
    oy += w * r * cosf(a-r);
""")

var('heart', """
    float sq = sqrtf(tx*tx + ty*ty);
    float a = sq * atan2f(tx, ty);
    float r = w * sq;
    ox += r * sinf(a);
    oy -= r * cosf(a);
""")

var('disc', """
    float a = w * atan2f(tx, ty) * M_1_PI;
    float r = M_PI * sqrtf(tx*tx + ty*ty);
    ox += sinf(r) * a;
    oy += cosf(r) * a;
""")

var('spiral', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    float r1 = w / r;
    ox += r1 * (cosf(a) + sinf(r));
    oy += r1 * (sinf(a) - cosf(r));
""")

var('hyperbolic', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    ox += w * sinf(a) / r;
    oy += w * cosf(a) * r;
""")

var('diamond', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    ox += w * sinf(a) * cosf(r);
    oy += w * cosf(a) * sinf(r);
""")

var('ex', """
    float a = atan2f(tx, ty);
    float r = sqrtf(tx*tx + ty*ty);
    float n0 = sinf(a+r);
    float n1 = cosf(a-r);
    float m0 = n0*n0*n0*r;
    float m1 = n1*n1*n1*r;
    ox += w * (m0 + m1);
    oy += w * (m0 - m1);
""")

var('julia', """
    float a = 0.5f * atan2f(tx, ty);
    if (mwc_next(rctx) & 1) a += M_PI;
    float r = w * sqrtf(sqrtf(tx*tx + ty*ty)); // TODO: fastest?
    ox += r * cosf(a);
    oy += r * sinf(a);
""")

var('bent', """
    float nx = 1.0f;
    if (tx < 0.0f) nx = 2.0f;
    float ny = 1.0f;
    if (ty < 0.0f) ny = 0.5f;
    ox += w * nx * tx;
    oy += w * ny * ty;
""")

var('waves', """
    float c10 = {{px.pre_affine.xy}};
    float c11 = {{px.pre_affine.yy}};

    ox += w * (tx + c10 * sinf(ty * {{pv.dx2}}));
    oy += w * (ty + c11 * sinf(tx * {{pv.dy2}}));
""", """
    float dx = {{px.pre_affine.offset.x}};
    float dy = {{px.pre_affine.offset.y}};
    {{pv._set('dx2')}} = 1.0f / (dx * dx + 1.0e-20f);
    {{pv._set('dy2')}} = 1.0f / (dy * dy + 1.0e-20f);
""")

var('fisheye', """
    float r = sqrtf(tx*tx + ty*ty);
    r = 2.0f * w / (r + 1.0f);
    ox += r * ty;
    oy += r * tx;
""")

var('popcorn', """
    float dx = tanf(3.0f*ty);
    float dy = tanf(3.0f*tx);
    ox += w * (tx + {{px.pre_affine.xo}} * sinf(dx));
    oy += w * (ty + {{px.pre_affine.yo}} * sinf(dy));
""")

var('exponential', """
    float dx = w * expf(tx - 1.0f);
    if (isfinite(dx)) {
        float dy = M_PI * ty;
        ox += dx * cosf(dy);
        oy += dx * sinf(dy);
    }
""")

var('power', """
    float a = atan2f(tx, ty);
    float sa = sinf(a);
    float r = w * powf(sqrtf(tx*tx + ty*ty),sa);
    ox += r * cosf(a);
    oy += r * sa;
""")

var('cosine', """
    float a = M_PI * tx;
    ox += w * cosf(a) * coshf(ty);
    oy -= w * sinf(a) * sinhf(ty);
""")

var('rings', """
    float dx = {{px.pre_affine.xo}};
    dx *= dx;
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    r = w * (fmodf(r+dx, 2.0f*dx) - dx + r * (1.0f - dx));
    ox += r * cosf(a);
    oy += r * sinf(a);
""")

var('fan', """
    float dx = {{px.pre_affine.xo}};
    dx *= dx * M_PI;
    float dx2 = 0.5f * dx;
    float dy = {{px.pre_affine.yo}};
    float a = atan2f(tx, ty);
    a += (fmodf(a+dy, dx) > dx2) ? -dx2 : dx2;
    float r = w * sqrtf(tx*tx + ty*ty);
    ox += r * cosf(a);
    oy += r * sinf(a);
""")

var('blob', """
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    float bdiff = 0.5f * ({{pv.high}} - {{pv.low}});
    r *= w * ({{pv.low}} + bdiff * (1.0f + sinf({{pv.waves}} * a)));
    ox += sinf(a) * r;
    oy += cosf(a) * r;
""")

var('pdj', """
    float nx1 = cosf({{pv.b}} * tx);
    float nx2 = sinf({{pv.c}} * tx);
    float ny1 = sinf({{pv.a}} * ty);
    float ny2 = cosf({{pv.d}} * ty);
    ox += w * (ny1 - nx1);
    oy += w * (nx2 - ny2);
""")

var('fan2', """
    float dy = {{pv.y}};
    float dx = {{pv.x}};
    dx *= dx * M_PI;
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
""")

var('rings2', """
    float dx = {{pv.val}};
    dx *= dx;
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    r += -2.0f * dx * (int)((r+dx)/(2.0f*dx)) + r * (1.0f - dx);
    ox += w * sinf(a) * r;
    oy += w * cosf(a) * r;
""")

var('eyefish', """
    float r = 2.0f * w / (sqrtf(tx*tx + ty*ty) + 1.0f);
    ox += r * tx;
    oy += r * ty;
""")

var('bubble', """
    float r = w / (0.25f * (tx*tx + ty*ty) + 1.0f);
    ox += r * tx;
    oy += r * ty;
""")

var('cylinder', """
    ox += w * sinf(tx);
    oy += w * ty;
""")

var('perspective', """
    float t = 1.0f / ({{pv.mdist}} - ty * {{pv.sin}});
    ox += w * {{pv.mdist}} * tx * t;
    oy += w * {{pv.cos}} * ty * t;
""", """
    float pang = {{pv.angle}} * M_PI_2;
    float pdist = fmaxf(1e-9, {{pv.dist}});
    {{pv._set('mdist')}} = pdist;
    {{pv._set('sin')}} = sin(pang);
    {{pv._set('cos')}} = pdist * cos(pang);
""")

var('noise', """
    float tmpr = mwc_next_01(rctx) * 2.0f * M_PI;
    float r = w * mwc_next_01(rctx);
    ox += tx * r * cosf(tmpr);
    oy += ty * r * sinf(tmpr);
""")

var('julian', """
    float power = {{pv.power}};
    float t_rnd = truncf(mwc_next_01(rctx) * fabsf(power));
    float a = atan2f(ty, tx);
    float tmpr = (a + 2.0f * M_PI * t_rnd) / power;
    float cn = {{pv.cn}};
    float r = w * powf(tx * tx + ty * ty, cn);

    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
""", """
    {{pv._set('cn')}} = {{pv.dist}} / (2.0f * {{pv.power}});
""")

var('juliascope', """
    float ang = atan2f(ty, tx);
    float power = {{pv.power}};
    float t_rnd = truncf(mwc_next_01(rctx) * fabsf(power));
    // TODO: don't draw the extra random number
    if (mwc_next(rctx) & 1) ang = -ang;
    float tmpr = (2.0f * M_PI * t_rnd + ang) / power;
    float r = w * powf(tx * tx + ty * ty, {{pv.cn}});

    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
""", """
    {{pv._set('cn')}} = {{pv.dist}} / (2.0f * {{pv.power}});
""")

var('blur', """
    float tmpr = mwc_next_01(rctx) * 2.0f * M_PI;
    float r = w * mwc_next_01(rctx);
    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
""")

var('gaussian_blur', """
    float ang = mwc_next_01(rctx) * 2.0f * M_PI;
    // constant factor here is stdev correction for converting to Box-Muller;
    // np.std(np.sum(np.random.random((1<<30, 4)), axis=1) - 2)
    // TODO: maybe derive it analytically
    float r = w * 0.57736 * sqrtf(-2.0f * log2f(mwc_next_01(rctx)) / M_LOG2E);
    ox += r * cosf(ang);
    oy += r * sinf(ang);
""")

var('radial_blur', """
    float blur_angle = {{pv.angle}} * M_PI * 0.5f;
    float spinvar = sinf(blur_angle);
    float zoomvar = cosf(blur_angle);

    float r = w * 0.57736 * sqrtf(-2.0f * log2f(mwc_next_01(rctx)) / M_LOG2E);
    float ra = sqrtf(tx*tx + ty*ty);
    float tmpa = atan2f(ty, tx) + spinvar * r;
    float rz = zoomvar * r - 1.0f;
    ox += ra*cosf(tmpa) + rz*tx;
    oy += ra*sinf(tmpa) + rz*ty;
""")

var('pie', """
    float slices = {{pv.slices}};
    float sl = truncf(mwc_next_01(rctx) * slices + 0.5f);
    float a = {{pv.rotation}} +
                2.0f * M_PI * (sl + mwc_next_01(rctx) * {{pv.thickness}}) / slices;
    float r = w * mwc_next_01(rctx);
    ox += r * cosf(a);
    oy += r * sinf(a);
""")

var('ngon', """
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
""")

var('curl', """
    float c1 = {{pv.c1}};
    float c2 = {{pv.c2}};

    float re = 1.0f + c1*tx + c2*(tx*tx - ty*ty);
    float im = c1*ty + 2.0f*c2*tx*ty;
    float r = w / (re*re + im*im);

    ox += r * (tx*re + ty*im);
    oy += r * (ty*re - tx*im);
""")

var('rectangles', """
    float rx = {{pv.x}};
    float ry = {{pv.y}};

    ox += w * ( (rx==0.0f) ? tx : rx * (2.0f * floorf(tx/rx) + 1.0f) - tx);
    oy += w * ( (ry==0.0f) ? ty : ry * (2.0f * floorf(ty/ry) + 1.0f) - ty);
""")

var('arch', """
    float ang = mwc_next_01(rctx) * w * M_PI;

    ox += w * sinf(ang);
    oy += w * sinf(ang) * sinf(ang) / cosf(ang);
""")

var('tangent', """
    ox += w * sinf(tx) / cosf(ty);
    oy += w * tanf(ty);
""")

var('square', """
    ox += w * (mwc_next_01(rctx) - 0.5f);
    oy += w * (mwc_next_01(rctx) - 0.5f);
""")

var('rays', """
    float ang = w * mwc_next_01(rctx) * M_PI;
    float r = w / (tx*tx + ty*ty);
    float tanr = w * tanf(ang) * r;
    ox += tanr * cosf(tx);
    oy += tanr * sinf(ty);
""")

var('blade', """
    float r = mwc_next_01(rctx) * w * sqrtf(tx*tx + ty*ty);
    ox += w * tx * (cosf(r) + sinf(r));
    oy += w * tx * (cosf(r) - sinf(r));
""")

var('secant2', """
    float r = w * sqrtf(tx*tx + ty*ty);
    float cr = cosf(r);
    float icr = 1.0f / cr;
    icr += (cr < 0 ? 1 : -1);

    ox += w * tx;
    oy += w * icr;
""")

# var 47 is twintrian, has a call to badvalue in it

var('cross', """
    float s = tx*tx - ty*ty;
    float r = w * sqrtf(1.0f / (s*s));

    ox += r * tx;
    oy += r * ty;
""")

var('disc2', """
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
""")

var('super_shape', """
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
""")

var('flower', """
    float holes = {{pv.holes}};
    float petals = {{pv.petals}};

    float r = w * (mwc_next_01(rctx) - holes)
                * cosf(petals*atan2f(ty, tx)) / sqrtf(tx*tx + ty*ty);

    ox += r * tx;
    oy += r * ty;
""")

var('conic', """
    float d = sqrtf(tx*tx + ty*ty);
    float ct = tx / d;
    float holes = {{pv.holes}};
    float eccen = {{pv.eccentricity}};

    float r = w * (mwc_next_01(rctx) - holes) * eccen / (1.0f + eccen*ct) / d;

    ox += r * tx;
    oy += r * ty;
""")

var('parabola', """
    float r = sqrtf(tx*tx + ty*ty);
    float sr = sinf(r);
    float cr = cosf(r);

    ox += {{pv.height}} * w * sr * sr * mwc_next_01(rctx);
    oy += {{pv.width}}  * w * cr * mwc_next_01(rctx);
""")

var('bent2', """
    float nx = 1.0f;
    if (tx < 0.0f) nx = {{pv.x}};
    float ny = 1.0f;
    if (ty < 0.0f) ny = {{pv.y}};
    ox += w * nx * tx;
    oy += w * ny * ty;
""")

var('bipolar', """
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
""")

var('boarders', """
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

var('butterfly', """
    /* wx is weight*4/sqrt(3*pi) */
    float wx = w * 1.3029400317411197908970256609023f;
    float y2 = ty * 2.0f;
    float r = wx * sqrtf(fabsf(ty * tx)/(tx*tx + y2*y2));
    ox += r * tx;
    oy += r * y2;
""")

var('cell', """
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
""")

var('cpow', """
    float a = atan2f(ty, tx);
    float lnr = 0.5f * logf(tx*tx+ty*ty);
    float power = 1.0f / {{pv.power}};
    float va = 2.0f * M_PI * power;
    float vc = {{pv.r}} * power;
    float vd = {{pv.i}} * power;
    float ang = vc*a + vd*lnr + va*floorf(power*mwc_next_01(rctx));
    float m = w * expf(vc * lnr - vd * a);
    ox += m * cosf(ang);
    oy += m * sinf(ang);
""")

var('curve', """
    float pc_xlen = {{pv.x2}}, pc_ylen = {{pv.y2}};

    ox += w * (tx + {{pv.xamp}} * expf(-ty*ty*pc_xlen));
    oy += w * (ty + {{pv.yamp}} * expf(-tx*tx*pc_ylen));
""", """
    float xl = {{pv.xlength}}, yl = {{pv.ylength}};
    {{pv._set('x2')}} = 1.0f / max(1e-20f, xl * xl);
    {{pv._set('y2')}} = 1.0f / max(1e-20f, yl * yl);
""")

var('edisc', """
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

var('elliptic', """
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

var('escher', """
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
""")

var('foci', """
    float expx = expf(tx) * 0.5f;
    float expnx = 0.25f / expx;
    float sn = sinf(ty);
    float cn = cosf(ty);
    float tmp = w / (expx + expnx - cn);
    ox += tmp * (expx - expnx);
    oy += tmp * sn;
""")

var('lazysusan', """
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
""")

var('loonie', """
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

var('pre_blur', """
    float rndG = w * (mwc_next_01(rctx) + mwc_next_01(rctx)
                   + mwc_next_01(rctx) + mwc_next_01(rctx) - 2.0f);
    float rndA = mwc_next_01(rctx) * 2.0f * M_PI;

    /* Note: original coordinate changed */
    tx += rndG * cosf(rndA);
    ty += rndG * sinf(rndA);
""")

var('modulus', """
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
""")

var('oscope', """
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
""")

var('polar2', """
    float p2v = w / M_PI;
    ox += p2v * atan2f(tx,ty);
    oy += 0.5f * p2v * logf(tx*tx + ty*ty);
""")

var('popcorn2', """
    float c = {{pv.c}};
    ox += w * (tx + {{pv.x}} * sinf(tanf(ty*c)));
    oy += w * (ty + {{pv.y}} * sinf(tanf(tx*c)));
""")

var('scry', """
    /* note that scry does not multiply by weight, but as the */
    /* values still approach 0 as the weight approaches 0, it */
    /* should be ok                                           */
    float t = tx*tx + ty*ty;
    float r = 1.0f / (sqrtf(t) * (t + 1.0f/w));
    ox += tx*r;
    oy += ty*r;
""")

var('separation', """
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
""")

var('split', """
    if (cosf(tx*{{pv.xsize}}*M_PI) >= 0.0f)
        oy += w*ty;
    else
        oy -= w*ty;

    if (cosf(ty*{{pv.ysize}}*M_PI) >= 0.0f)
        ox += w*tx;
    else
        ox -= w*tx;
""")

var('splits', """
    ox += w*(tx + copysignf({{pv.x}}, tx));
    oy += w*(ty + copysignf({{pv.y}}, ty));
""")

var('stripes', """
    float roundx = floorf(tx + 0.5f);
    float offsetx = tx - roundx;
    ox += w * (offsetx * (1.0f - {{pv.space}}) + roundx);
    oy += w * (ty + offsetx*offsetx*{{pv.warp}});
""")

var('wedge', """
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
""")

var('whorl', """
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(ty,tx);
    
    if (r < w)
       a += {{pv.inside}} / (w - r);
    else
       a += {{pv.outside}} / (w - r);

    ox += w * r * cosf(a);
    oy += w * r * sinf(a);
""")
    
var('waves2', """
    ox += w*(tx + {{pv.scalex}}*sinf(ty * {{pv.freqx}}));
    oy += w*(ty + {{pv.scaley}}*sinf(tx * {{pv.freqy}}));
""")

var('exp', """
    float expe = expf(tx);
    ox += w * expe * cosf(ty);
    oy += w * expe * sinf(ty);
""")

var('log', """
    ox += w * 0.5f * logf(tx*tx + ty*ty);
    oy += w * atan2f(ty, tx);
""")

var('sin', """
    ox += w * sinf(tx) * coshf(ty);
    oy += w * cosf(tx) * sinhf(ty);
""")

var('cos', """
    ox += w * cosf(tx) * coshf(ty);
    oy -= w * sinf(tx) * sinhf(ty);
""")

var('tan', """
    float tanden = 1.0f/(cosf(2.0f*tx) + coshf(2.0f*ty));
    ox += w * tanden * sinf(2.0f*tx);
    oy += w * tanden * sinhf(2.0f*ty);
""")

var('sec', """
    float secden = 2.0f/(cosf(2.0f*tx) + coshf(2.0f*ty));
    ox += w * secden * cosf(tx) * coshf(ty);
    oy += w * secden * sinf(tx) * sinhf(ty);
""")

var('csc', """
    float cscden = 2.0f/(coshf(2.0f*ty) - cosf(2.0f*tx));
    ox += w * cscden * sinf(tx) * coshf(ty);
    oy -= w * cscden * cosf(tx) * sinhf(ty);
""")

var('cot', """
    float cotden = 1.0f/(coshf(2.0f*ty) - cosf(2.0f*tx));
    ox += w * cotden * sinf(2.0f*tx);
    oy += w * cotden * -1.0f * sinhf(2.0f*ty);
""")

var('sinh', """
    ox += w * sinhf(tx) * cosf(ty);
    oy += w * coshf(tx) * sinf(ty);
""")

var('cosh', """
    ox += w * coshf(tx) * cosf(ty);
    oy += w * sinhf(tx) * sinf(ty);
""")

var('tanh', """
    float tanhden = 1.0f/(cosf(2.0f*ty) + coshf(2.0f*tx));
    ox += w * tanhden * sinhf(2.0f*tx);
    oy += w * tanhden * sinf(2.0f*ty);
""")

var('sech', """
    float sechden = 2.0f/(cosf(2.0f*ty) + coshf(2.0f*tx));
    ox += w * sechden * cosf(ty) * coshf(tx);
    oy -= w * sechden * sinf(ty) * sinhf(tx);
""")

var('csch', """
    float cschden = 2.0f/(coshf(2.0f*tx) - cosf(2.0f*ty));
    ox += w * cschden * sinhf(tx) * cosf(ty);
    oy -= w * cschden * coshf(tx) * sinf(ty);
""")

var('coth', """
    float cothden = 1.0f/(coshf(2.0f*tx) - cosf(2.0f*ty));
    ox += w * cothden * sinhf(2.0f*tx);
    oy += w * cothden * sinf(2.0f*ty);
""")

var('flux', """
    float xpw = tx + w;
    float xmw = tx - w;
    float avgr = w * (2.0f + {{pv.spread}})
               * sqrtf(sqrtf(ty*ty+xpw*xpw)/sqrtf(ty*ty+xmw*xmw));
    float avga = (atan2f(ty, xmw) - atan2f(ty,xpw))*0.5f;
    ox += avgr * cosf(avga);
    oy += avgr * sinf(avga);
""")

var('mobius', """
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
""")
