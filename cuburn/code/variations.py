
var_nos = {}
var_code = {}

def _var(num, name = None):
    def _var1(func):
        if name:
            namelo = name
        else:
            namelo = func.__name__
        var_nos[num] = namelo
        var_code[namelo] = func
    return _var1

# Variables note: all functions will have their weights as 'w',
# input variables 'tx' and 'ty', and output 'ox' and 'oy' available
# from the calling context. Each statement will be placed inside brackets,
# to avoid namespace pollution.
@_var(0)
def linear():
    return """
    ox += tx * w;
    oy += ty * w;
    """

@_var(1)
def sinusoidal():
    return """
    ox += w * sin(tx);
    oy += w * sin(ty);
    """

@_var(2)
def spherical():
    return """
    float r2 = w / (tx*tx + ty*ty);
    ox += tx * r2;
    oy += ty * r2;
    """

