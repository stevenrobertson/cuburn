import base64
import numpy as np

from cuburn.code.util import crep

def get(dct, default, *keys):
    if len(keys) == 1:
        keys = keys[0].split('.')
    for k in keys:
        if k in dct:
            dct = dct[k]
        else:
            return default
    return dct

def flatten(dct, ctx=()):
    """
    Given a nested dict, return a flattened dict with dot-separated string
    keys. Keys that have dots in them already are treated the same.

    >>> flatten({'ab': {'xy.zw': 1}, 4: 5}) == {'ab.xy.zw': 1, '4': 5}
    True
    """

    for k, v in dct.items():
        k = str(k)
        if isinstance(v, dict):
            for sk, sv in flatten(v, ctx + (k,)):
                yield sk, sv
        else:
            yield '.'.join(ctx + (k,)), v

def unflatten(kvlist):
    """
    Given a flattened dict, return a nested dict, where every dot-separated
    key is converted into a sub-dict.

    >>> (unflatten([('ab.xy.zw', 1), ('4', 5)) ==
    ...  {'ab': {'xy': {'zw': 1}}, '4': 5})
    True
    """
    def go(d, k, v):
        if len(k) == 1:
            d[k[0]] = v
        else:
            go(d.setdefault(k[0], {}), k[1:], v)
    out = {}
    for k, v in kvlist:
        go(out, k.split('.'), v)
    return out


def palette_decode(datastrs):
    """
    Decode a palette (stored as a list suitable for JSON packing) into a
    palette. Internal palette format is simply as a (256,4) array of [0,1]
    RGBA floats.
    """
    if datastrs[0] != 'rgb8':
        raise NotImplementedError
    raw = base64.b64decode(''.join(datastrs[1:]))
    pal = np.reshape(np.fromstring(raw, np.uint8), (256, 3))
    data = np.ones((256, 4), np.float32)
    data[:,:3] = pal / 255.0
    return data

def palette_encode(data, format='rgb8'):
    """
    Encode an internal-format palette to an external representation.
    """
    if format != 'rgb8':
        raise NotImplementedError
    clamp = np.maximum(0, np.minimum(255, np.round(data[:,:3]*255.0)))
    enc = base64.b64encode(np.uint8(clamp))
    return ['rgb8'] + [enc[i:i+64] for i in range(0, len(enc), 64)]

def json_encode(obj):
    """
    Encode an object into JSON notation, formatted to be more readable than
    the output of the standard 'json' package for genomes.

    This serializer only works on the subset of JSON used in genomes.
    """
    result = _js_enc_obj(obj).lstrip()
    result = '\n'.join(l.rstrip() for l in result.split('\n'))
    return result + '\n'

def _js_enc_obj(obj, indent=0):
    isnum = lambda v: isinstance(v, (float, int, np.number))

    def wrap(pairs, delims):
        do, dc = delims
        i = ' ' * indent
        out = ''.join([do, ', '.join(pairs), dc])
        if '\n' not in out and len(out) + indent < 70:
            return out
        return ''.join(['\n', i, do, ' ', ('\n'+i+', ').join(pairs),
                        '\n', i, dc])

    if isinstance(obj, dict):
        if not obj:
            return '{}'
        digsort = lambda kv: (int(kv[0]), kv[1]) if kv[0].isdigit() else kv
        ks, vs = zip(*sorted(obj.items(), key=digsort))
        if ks == ('b', 'g', 'r'):
            ks, vs = reversed(ks), reversed(vs)
        ks = [crep('%.6g' % k if isnum(k) else str(k)) for k in ks]
        vs = [_js_enc_obj(v, indent+2) for v in vs]
        return wrap(['%s: %s' % p for p in zip(ks, vs)], '{}')
    elif isinstance(obj, list):
        vs = [_js_enc_obj(v, indent+2) for v in obj]
        if vs and len(vs) % 2 == 0 and isnum(obj[1]):
            vs = map(', '.join, zip(vs[::2], vs[1::2]))
        return wrap(vs, '[]')
    #elif isinstance(obj, SplEval):
        #return _js_enc_obj(obj.knotlist, indent)
    elif isinstance(obj, basestring):
        return crep(obj)
    elif isnum(obj):
        return '%.6g' % obj
    raise TypeError("Don't know how to serialize %s of type %s" %
                    (obj, type(obj)))
