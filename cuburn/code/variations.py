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
    ox += r * sinf(a)
    oy -= r * cosf(a)
    """)

var(8, 'disc', """
    float a = w * atan2f(tx, ty) * M_1_PI;
    float r = M_PI * sqrtf(tx*tx + ty*ty);
    ox += sinf(r) * a
    oy += cosf(r) * a 
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
    float a = 0.5f * atan2f(tx, ty)
    if (mwc_next(rctx) & 1) a += M_PI;
    float r = w * sqrtf(tx*tx + ty*ty);
    ox += r * cosf(a);
    oy += r * sinf(a);
    """)

var(14, 'bent', """
    float nx = 1.0f;
    if (tx < 0.0f) nx = 2.0f;
    float ny = 1.0;
    if (ty < 0.0f) ty = 0.5f;
    ox += w * nx * tx;
    oy += w * ny * ty;
    """)

var(15, 'waves', """
    float c10 = {{px.get(None, 'pre_xy')}};
    float c11 = {{px.get(None, 'pre_yy')}};
    ox += w * (tx + c10 + sinf(ty * {{px.get('xf.waves_dx2')}}));
    oy += w * (ty + c11 + sinf(tx * {{px.get('xf.waves_dy2')}}));
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
    ox += w * (tx + {{px.get(None, 'pre_xo')}} * sinf(dx));
    oy += w * (ty + {{px.get(None, 'pre_yo')}} * sinf(dy));
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
    float dx = {{px.get(None, 'pre_xo')}} * {{px.get(None, 'pre_xo')}};
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    r = w * (fmodf(r+dx, 2.0f*dx) - dx + r * (1.0f - dx));
    ox += r * cosf(a);
    oy += r * sinf(a);
    """)

var(22, 'fan', """
    float dx = M_PI * ({{px.get(None, 'pre_xo')}} * {{px.get(None, 'pre_xo')}});
    float dx2 = 0.5f * dx;
    float dy = {{px.get(None, 'pre_yo')}};
    float a = atan2f(tx, ty);
    a += (fmodf(a+dy, dx) > dx2) ? -dx2 : dx2;
    float r = w * sqrtf(tx*tx + ty*ty);
    ox += r * cosf(a);
    oy += r * sinf(a);
    """)

var(23, 'blob', """
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    float bdiff = 0.5f * ({{px.get('xf.blob_high - xf.blob_low','blob_diff'}})
    r *= w * ({{px.get('xf.blob_low')}} + bdiff * (1.0f + sinf({{px.get('xf.blob_waves')}} * a)))
    ox += sinf(a) * r;
    oy += cosf(a) * r;
    """)

var(24, 'pdj', """
    float nx1 = cosf({{px.get('xf.pdj_b')}} * tx);
    float nx2 = sinf({{px.get('xf.pdj_c')}} * tx);
    float ny1 = sinf({{px.get('xf.pdj_a')}} * ty);
    float ny2 = cosf({{px.get('xf.pdj_d')}} * ty);
    ox += w * (ny1 - nx1);
    oy += w * (nx2 - ny2);
    """)

var(25, 'fan2', """
    float dy = {{px.get('xf.fan2_y')}};
    float dx = M_PI * {{px.get('xf.fan2_x')}} * {{px.get('xf.fan2_x')}};
    float dx2 = 0.5f * dx;
    float a = atan2f(tx, ty);
    float r = w * sqrtf(tx*tx + ty*ty);
    if (t > dx2)
        a -= dx2;
    else
        a += dx2;
    
    ox += r * sinf(a);
    oy += r * cosf(a);
    """)

var(26, 'rings2', """
    float dx = {{px.get('xf.rings2_val')}} * {{px.get('xf.rings2_val')}};
    float r = sqrtf(tx*tx + ty*ty);
    float a = atan2f(tx, ty);
    r += -2.0f * dx * (int)((r+dx)/(2.0f*dx)) + r * (1.0f - dx);
    ox += w * sinf(a) * r;
    oy += w * cosf(a) * r;
    """)

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

