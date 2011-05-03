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
    float r2 = w / (tx*tx + ty*ty + 1e-20);
    ox += tx * r2;
    oy += ty * r2;
    """)

var(5, 'polar', """
    ox += w * atan2f(tx, ty) * M_1_PI;
    oy += w * (sqrtf(tx * tx + ty * ty) - 1.0);
    """)


var(10, 'hyperbolic', """
    float a = atan2f(tx, ty);
    float r = sqrt(tx*tx + ty*ty) + 1e-20;
    ox += w * sinf(a) / r;
    oy += w * cosf(a) * r;
    """)

var(33, 'juliascope', """
    float ang = atan2f(ty, tx);
    float power = {{px.get('xf.juliascope_power', 'juscope_power')}};
    float t_rnd = truncf(mwc_next_01(rctx) * fabsf(power));
    // TODO: don't draw the extra random number
    if (mwc_next(rctx) & 1) ang = -ang;
    float tmpr = (2 * M_PI * t_rnd + ang) / power;

    float cn = {{px.get('xf.juliascope_dist / xf.juliascope_power / 2',
                         'juscope_cn')}};
    float r = w * powf(tx * tx + ty * ty, cn);

    ox += r * cosf(tmpr);
    oy += r * sinf(tmpr);
    """)

