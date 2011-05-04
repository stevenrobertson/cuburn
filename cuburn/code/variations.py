import tempita

var_nos = {}
var_code = {}

def var(num, name, code):
    var_nos[num] = name
    var_code[name] = tempita.Template(code)

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
    float r2 = w / (tx*tx + ty*ty + 1e-20f);
    ox += tx * r2;
    oy += ty * r2;
    """)

var(3, 'swirl', """
    float r2 = tx*tx + ty*ty;
    float c1 = sin(r2);
    float c2 = cos(r2);
    ox += w * (c1*tx - c2*ty);
    oy += w * (c2*tx + c1*ty);
    """)

var(4, 'horseshoe', """
    float r = w / sqrt(tx*tx + ty*ty);
    ox += r * (tx - ty) * (tx + ty);
    oy += 2.0 * tx * ty * r;
    """)

var(5, 'polar', """
    ox += w * atan2f(tx, ty) * M_1_PI;
    oy += w * (sqrtf(tx * tx + ty * ty) - 1.0f);
    """)

var(6, 'handkerchief', """
    float a = atan2f(tx, ty);
    float r = sqrt(tx*tx + ty*ty);
    ox += w * r * sin(a+r);
    oy += w * r * cos(a-r);
    """)

var(7, 'heart', """
    float sq = sqrt(tx*tx + ty*ty);
    float a = sq * atan2f(tx, ty);
    float r = w * sq;
    ox += r * sin(a)
    oy -= r * cos(a)
    """)

var(8, 'disc', """
    float a = w * atan2f(tx, ty) * M_1_PI;
    float r = M_PI * sqrt(tx*tx + ty*ty);
    ox += sin(r) * a
    oy += cos(r) * a 
    """)

var(9, 'spiral', """
    float a = atan2f(tx, ty);
    float r = sqrt(tx*tx + ty*ty);
    float r1 = w / r;
    ox += r1 * (cos(a) + sin(r));
    oy += r1 * (sin(a) - cos(r));
    """)

var(10, 'hyperbolic', """
    float a = atan2f(tx, ty);
    float r = sqrt(tx*tx + ty*ty) + 1e-20f;
    ox += w * sinf(a) / r;
    oy += w * cosf(a) * r;
    """)

var(11, 'diamond', """
    float a = atan2f(tx, ty);
    float r = sqrt(tx*tx + ty*ty);
    ox += w * sin(a) * cos(r);
    oy += w * cos(a) * sin(r);
    """)

var(12, 'ex', """
    float a = atan2f(tx, ty);
    float r = sqrt(tx*tx + ty*ty);
    float n0 = sin(a+r);
    float n1 = cos(a-r);
    float m0 = n0*n0*n0*r;
    float m1 = n1*n1*n1*r;
    ox += w * (m0 + m1);
    oy += w * (m0 - m1);
    """)

var(13, 'julia', """
    float a = 0.5 * atan2f(tx, ty)
    if (mwc_next(rctx) & 1) a += M_PI;
    float r = w * sqrt(tx*tx + ty*ty);
    ox += r * cos(a);
    oy += r * sin(a);
    """)

var(14, 'bent', """
    float nx = 1.0;
    if (tx < 0.0) nx = 2.0;
    float ny = 1.0;
    if (ty < 0.0) ty = 0.5;
    ox += w * nx * tx;
    oy += w * ny * ty;
    """)

var(15, 'waves', """
    float c10 = {{px.get(None, 'pre_yx')}};
    float c11 = {{px.get(None, 'pre_yy')}};
    ox += w * (tx + c10 + sin(ty * {{px.get('xf.waves_dx2')}}));
    oy += w * (ty + c11 + sin(tx * {{px.get('xf.waves_dy2')}}));
    """)

var(16, 'fisheye', """
    float r = sqrt(tx*tx + ty*ty);
    r = 2.0 * w / (r + 1.0);
    ox += r * ty;
    oy += r * tx;
    """)

var(33, 'juliascope', """
    float ang = atan2f(ty, tx);
    float power = {{px.get('xf.juliascope_power', 'juscope_power')}};
    float t_rnd = truncf(mwc_next_01(rctx) * fabsf(power));
    // TODO: don't draw the extra random number
    if (mwc_next(rctx) & 1) ang = -ang;
    float tmpr = (2.0f * M_PI * t_rnd + ang) / power;

    float cn = {{px.get('xf.juliascope_dist / xf.juliascope_power / 2',
                         'juscope_cn')}};
    float r = w * powf(tx * tx + ty * ty, cn);

    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
    """)

