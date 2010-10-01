"""
PTX DSL, a domain-specific language for NVIDIA's PTX.
"""

# If you see 'import inspect', you know you're in for a good time
import inspect
import struct
from cStringIO import StringIO
from collections import namedtuple
from math import *

import numpy as np
import pycuda.driver as cuda

from pprint import pprint

PTX_VERSION=(2, 1)

Type = namedtuple('Type', 'name kind bits bytes')
TYPES = {}
for kind in 'busf':
    for width in [8, 16, 32, 64]:
        TYPES[kind+str(width)] = Type(kind+str(width), kind, width, width / 8)
del TYPES['f8']
TYPES['pred'] = Type('pred', 'p', 0, 0)

class Statement(object):
    """
    Representation of a PTX statement.
    """
    known_opnames = ('add addc sub subc mul mad mul24 mad24 sad div rem abs '
        'neg min max popc clz bfind brev bfe bfi prmt testp copysign rcp '
        'sqrt rsqrt sin cos lg2 ex2 set setp selp slct and or xor not '
        'cnot shl shr mov ld ldu st prefetch prefetchu isspacep cvta cvt '
        'tex txq suld sust sured suq bra call ret exit bar membar atom red '
        'vote vadd vsub vabsdiff vmin vmax vshl vshr vmad vset').split()

    def __init__(self, name, args, line_info = None):
        self.opname = name
        self.fullname, self.operands, self.rtype = self.parse(name, args)
        self.result = None
        self.python_line = line_info
        self.ptx_line = None

    @staticmethod
    def parse(name, args):
        """
        Parses and expands a (possibly incomplete) PTX statement, returning the
        complete operation name and destination register type.

        ``name`` is a list of the parts of the operation name (as would be
                 given by ``'add.u32'.split()``, for example).
        ``args`` is a list of the arguments to the operation, excluding the
                 destination register.

        Returns a 3-tuple of ``(fullname, args, rtype)``, where ``fullname`` is
        the fully-expanded name of the operation, ``args`` is the list of
        arguments with all untyped values converted to ``Immediate`` values of
        the appropriate type, and ``type`` is the expected result type of the
        statement. If the statement does not have a destination register,
        ``type`` will be None.
        """
        # TODO: .ftz insertion

        if name[0] in 'tex txq suld sust sured suq call'.split():
            raise NotImplementedError("No support for %s yet" % name[0])

        # Make sure we don't modify the caller's list/tuple
        name, args = list(name), list(args)

        # Six constants that just have to be unique from each other
        # 'stype', 'dtype', 'ignore', 'u32', 'pred', 'memory'
        ST, DT, IG, U3, PR, ME = range(6)

        if name[0] in ('add addc sub subc mul mul24 div rem min max and or '
                       'xor not cnot copysign').split():
            atypes = [ST, ST]
        elif name[0] in ('abs neg popc clz bfind brev testp rcp sqrt rsqrt sin '
                         'cos lg2 ex2 mov cvt cvta isspacep split').split():
            atypes = [ST]
        elif name[0] == 'mad' and name[1] == 'wide':
            atypes = [ST, ST, DT]
        elif name[0] in 'mad mad24 sad'.split():
            atypes = [ST, ST, ST]
        elif name[0] == 'bfe':
            atypes = [ST, U3, U3]
        elif name[0] == 'bfi':
            atypes = [ST, ST, U3, U3]
        elif name[0] == 'prmt':
            atypes = [U3, U3, U3]
        elif name[0] in 'ld ldu prefetch prefetchu':
            atypes = [ME]
        elif name[0] == 'st':
            atypes = [ME, ST]
        elif name[0] in 'set setp selp'.split():
            atypes = [ST, ST, IG]
        elif name[0] == 'slct':
            atypes = [DT, DT, ST]
        elif name[0] in ('shl', 'shr'):
            atypes = [ST, U3]
        elif name[0] in ('atom', 'red'):
            if name[1] == 'cas':
                atypes = [ME, ST, ST]
            else:
                atypes = [ME, ST]
        elif name[0] in 'ret exit membar'.split():
            atypes = []
        elif name[0] == 'vote':
            atypes = [PR]
        elif name[0] in 'bar':
            atypes = [U3, IG, IG]
        else:
            raise NotImplementedError("Don't recognize the %s statement. "
                    "If you think this is a bug, and it may well be, please "
                    "report it!" % name[0])

        if (len(args) < len(filter(lambda t: t != IG, atypes)) or
            len(args) > len(atypes)):
            print args
            print atypes
            raise ValueError("Incorrect number of args for '%s'" % name[0])

        stype, dtype = None, None
        did_inference = False

        if isinstance(args[0], Pointer):
            # Get stype from pointer (explicit stype overrides this)
            if name[0] in 'ld ldu st'.split():
                stype = args[0].dtype
                did_inference = True
            # Get sspace from pointer if missing
            if name[0] in 'ld ldu st prefetch atom red'.split():
                sspos = 2 if len(name) > 1 and name[1] == 'volatile' else 1
                if (len(name) <= sspos or name[sspos] not in
                        'global local shared param const'.split()):
                    name.insert(sspos, args[0].sspace)

        # These instructions lack an stype suffix
        if name[0] in ('prmt prefetch prefetchu isspacep bra ret exit membar '
                       'bar vote'.split()):
            # False (as opposed to None) prevents stype inference attempt
            stype = False
        else:
            # These instructions require a dtype
            if name[0] in 'set slct cvt':
                if name[-1] not in TYPES:
                    raise SyntaxError("'%s' requires a dtype." % name[0])
                if name[-2] in TYPES:
                    dtype, stype = TYPES[name[-2]], TYPES[name[-1]]
                else:
                    dtype = TYPES[name[-1]]
            else:
                if name[-1] in TYPES:
                    stype = TYPES[name[-1]]
                    did_inference = False

        # stype wasn't explicitly set, try to infer it from the arguments
        if stype is None:
            maybe_typed = [a for a, t in zip(args, atypes) if t == ST]
            types = [a.type for a in maybe_typed if isinstance(a, Register)]
            if not types:
                raise TypeError("Not enough information to infer type. "
                        "Explicitly specify the source argument type.")
            stype = types[0]
            did_inference = True

        if did_inference:
            name.append(stype.name)

        # These instructions require a 'b32'-type argument, despite working
        # on u32 and s32 types just fine, so change the name but not stype
        if name[0] in 'popc clz bfind brev bfi and or xor not cnot shl'.split():
            name[-1] = 'b' + name[1:]

        # Calculate destination type (may influence some args too)
        if (name[0] in 'popc clz bfind prmt'.split() or
                name[:3] == ['bar', 'red', 'popc'] or
                name[:2] == ['vote', 'ballot']):
            dtype = TYPES['u32']
        elif (name[0] in 'testp setp isspacep vote'.split() or
                name[:2] == ['bar', 'red']):
            dtype = TYPES['pred']
        elif (name[0] in 'st prefetch prefetchu bra ret exit bar membar '
                         'red'.split()):
            dtype = None
        elif name[0] in ('mul', 'mad') and name[1] == 'wide':
            dtype = TYPES[stype.kind + str(2*stype.bits)]
        elif dtype is None:
            dtype = stype

        atype_dict = {ST: stype, DT: dtype, U3: TYPES['u32']}

        # Wrap any untyped immediates
        for idx, arg in enumerate(args):
            if not isinstance(arg, Register):
                t = atype_dict.get(atypes[idx])
                args[idx] = Immediate(None, t, arg)

        if did_inference:
            for i, (arg, atype) in enumerate(zip(args, atypes)):
                if atype in atype_dict and arg.type != atype_dict[atype]:
                    raise TypeError("Arg %d differs from expected type %s. "
                        "If this is intentional, explicitly specify the "
                        "source argument type." % (i, atype.name))
            if name[0] in 'ld ldu st red atom'.split():
                if (isinstance(args[0], Pointer) and
                    args[0].dtype.bits != stype.bits):
                    raise TypeError("The inferred type %s differs in size "
                        "from the referent's type %s. If this is intentional, "
                        "explicitly specify the source argument type." %
                        (stype.name, args[0].dtype.name))

        return name, tuple(args), dtype

class Register(object):
    """
    The workhorse.
    """
    def __init__(self, entry, type):
        self.entry, self.type = entry, type
        # Ordinary register naming / lifetime tracking
        self.name, self.inferred_name, self.rebound_to = None, None, None
        # Immediate value binding and other non-user-exposed hackery
        self._ptx = None

    def _set_val(self, val):
        if not isinstance(val, Register):
            val = Immediate(self.entry, self.type, val)
        self.entry.add_rebinding(self, val)
    val = property(lambda s: s, _set_val)
    def __repr__(self):
        s = super(Register, self).__repr__()[:-1]
        return s + ': type=%s, name=%s, inferred_name=%s>' % (
                    self.type.name, self.name, self.inferred_name)
    def get_name(self):
        if self._ptx is not None:
            return str(self._ptx)
        if self.rebound_to:
            return self.rebound_to.get_name()
        return self.name or self.inferred_name

    def _infer_name(self, depth=2):
        """
        To produce more readable code, this method reaches in to the stack and
        tries to find the name of this register in the calling method's locals.
        If a register is still unbound at code generation time, this name will
        be preferred over a meaningless ``rXX``-style identifier.

        This best-guess effort should have absolutely no semantic impact on the
        generated PTX, and is only here for readability, so we don't sweat the
        potential corner cases associated with it.

        ``depth`` is the index of the relevant frame in this function's stack.
        """
        if self.inferred_name is None:
            frame = inspect.stack()[depth][0]
            for key, val in frame.f_locals.items():
                if self is val:
                    self.inferred_name = key
                    break

class Pointer(Register):
    """
    A register which knows (in Python, at least) the type, state space, and
    address of a datum in memory.
    """
    # TODO: use u64 as type if device has >=4GB of memory
    ptr_type = TYPES['u32']
    def __init__(self, entry, sspace, dtype):
        super(Pointer, self).__init__(entry, self.ptr_type)
        self.sspace, self.dtype = sspace, dtype

class Immediate(Register):
    """
    An Immediate is the DSL's way of storing PTX immediate values. It differs
    from a Register in two respects:

    - A non-Register value can be assigned to the ``val`` property (or passed
      to ``__init__``). If the value is an int or float, it will be coerced to
      follow PTX's strict parsing rules for the type of the ``Immediate``;
      otherwise, it'll simply be coerced to ``str`` and pasted in the PTX.

    - The ``type`` can be None, which disables all coercion and introspection.
      This is practical for labels and the like.
    """
    def __init__(self, entry, type, val=None):
        super(Immediate, self).__init__(entry, type)
        self.val = val
    def _set_val(self, val):
        self._ptx = self.coerce(self.type, val)
    val = property(lambda s: s._ptx, _set_val)
    def __repr__(self):
        return object.__repr__(self)[:-1] + ': type=%s, value=%s>' % (
                self.type.name, self._ptx)
    @staticmethod
    def coerce(type, val):
        if type is None or not isinstance(val, (int, long, float)):
            return val
        if type.kind == 'u' and val < 0:
            raise ValueError("Can't convert (< 0) val to unsigned")
        # Maybe more later?
        if type.kind in 'us':
            return int(val)
        if type.kind in 'f':
            return float(val)
        raise TypeError("Immediates not supported for type %s" % type.name)

class Regs(object):
    """
    The ``entry.regs`` object to which Registers are bound.
    """
    def __init__(self, entry):
        self.__dict__['_entry'] = entry
        self.__dict__['_named_regs'] = dict()
    def __create_register_func(self, type):
        def f(*args, **kwargs):
            return self._entry.create_register(type, *args, **kwargs)
        return f
    def __getattr__(self, name):
        if name in TYPES:
            return self.__create_register_func(TYPES[name])
        if name in self._named_regs:
            return self._named_regs[name]
        raise KeyError("Unrecognized register name %s" % name)
    def __setattr__(self, name, val):
        if name in self._named_regs:
            self._named_regs[name].val = val
        else:
            if isinstance(val, Register):
                assert val in self._entry._regs, "Reg from nowhere!"
                val.name = name
                self._named_regs[name] = val
            else:
                raise TypeError("What Is This %s You Have Given Me" % val)


class Memory(object):
    """
    Memory objects reference device memory and and provide a convenient
    shorthand for address calculations.

    The base address of a memory location may be retreived from the ``addr``
    property as a ``Pointer`` for manual address calculations.

    Somewhat more automatic address calculations can be performed using Python
    bracket notation::

        >>> r1 = o.ld(m.something[r2])
        >>> o.st(m.something[2*r2], r1)

    If the value passed in the brackets is u32, it will *not* be coerced to
    u64 until being added to the base pointer. To access arrays that are more
    than 4GB in size, you must coerce the input type to u64 yourself.

    Currently, all steps in an address calculation are performed for each
    access, and so for inner loops manual address calculation (or simply saving
    the resulting register for reuse in the next memory operation) may be more
    efficient. Once the register lifetime profiler is complete, that behavior
    may change.
    """
    def __init__(self, entry, space, type, name):
        self.entry, self.space, self.type, self.name = entry, space, type, name
    @property
    def addr(self):
        ptr = Pointer(self.entry, self.space, self.type)
        ptr._ptx = self.name
    def __getitem__(self, key):
        # TODO: make this multi-type-safe, perform strength reduction/precalc
        ptr = Pointer(self.entry, self.space, self.type)
        self.entry.add_stmt(['mad', 'lo', 'u32'], key, self.type.bytes,
                            self.addr, result=ptr)
        return ptr

class PtrParam(Memory):
    """
    Entry parameters which contain pointers to memory locations, as created
    through ``entry.add_ptr_param()``, use this type to hide the address load
    from parameter space.
    """
    # TODO: this assumes u32 addresses, which won't be true for long
    @property
    def addr(self):
        ptr = Pointer(self.entry, self.space, self.type)
        self.entry.add_stmt(['ld', 'param', ptr.type.name],
                            self.name, result=ptr)
        return ptr

class Params(object):
    """
    The ``entry.params`` object to which parameters are bound.
    """
    def __init__(self, entry):
        # Boy this 'everything references entry` thing has gotten old
        self.entry = entry
    def __getattr__(self, name):
        if name not in self.entry._params:
            raise KeyError("Did not recognize parameter name.")
        param = self.entry._params[name]
        if isinstance(param, PtrParam):
            return param
        return self.entry.ops.ld(param.addr)

class _DotNameHelper(object):
    def __init__(self, callback, name = ()):
        self.__callback = callback
        self.__name = name
    def __getattr__(self, name):
        return _DotNameHelper(self.__callback, self.__name + (name,))
    def __call__(self, *args, **kwargs):
        return self.__callback(self.__name, *args, **kwargs)

RegUse = namedtuple('RegUse', 'src dst')
Rebinding = namedtuple('Rebinding', 'dst src')

class Entry(object):
    """
    Manager extraordinaire.

    TODO: document this.
    """

    def __init__(self, name, block_width, block_height=1, block_depth=1):
        self.name = name
        self.block = (block_width, block_height, block_depth)
        self.threads_per_cta = block_width * block_height
        self.body_seen = False
        self.tail_cbs = []
        self.identifiers = set()

        self.ops = _DotNameHelper(self.add_stmt)
        self._stmts = []
        self._labels = []
        self.regs = Regs(self)
        self._regs = {}

        # Intended to be read by the ``params`` object below
        self._params = {}
        self.params = Params(self)

    def __enter__(self):
        # May do more later
        pass

    def __exit__(self, etype, eval, tb):
        # May do more later
        pass

    def add_stmt(self, name, *operands, **kwargs):
        stmt = Statement(name, operands)
        idx = len(self._stmts)
        for operand in stmt.operands:
            operand._infer_name(2)
            use = self._regs.setdefault(operand, RegUse([], []))
            use.src.append(idx)
        if stmt.rtype is not None:
            result = kwargs.pop('result', None)
            if result:
                assert result.type == stmt.rtype, "Internal type error"
            else:
                result = Register(self, stmt.rtype)
            stmt.result = result
            self._regs[result] = RegUse(src=[], dst=[idx])
        if kwargs:
            raise KeyError("Unrecognized keyword arguments: %s" % kwargs)
        self._stmts.append(stmt)
        return stmt.result

    def add_rebinding(self, dst, src):
        idx = len(self._stmts)
        self._regs[dst].dst.append(idx)
        if not isinstance(src, Immediate):
            self._regs[src].src.append(idx)
        self._stmts.append(Rebinding(dst, src))

    def create_register(self, type, initial=None):
        r = Register(self, type)
        self._regs[r] = RegUse([], [])
        if initial:
            r.val = initial
        return r

    def head(self):
        """
        Top-level code segment that will be placed at the start of the entry.
        Useful for initialization of memory or registers by types that do
        not implement an entry point themselves.
        """
        # This may do more later
        return self

    def body(self):
        """
        Top-level code segment representing the body of the entry point.
        """
        # This may do more later
        assert not self.body_seen, "Only one body per entry allowed."
        self.body_seen = True
        return self

    def tail_callback(self, cb, *args, **kwargs):
        """
        Registers a tail callback function. After the body segment is complete,
        the tail callbacks will be called in reverse, such that each head/tail
        pair nests in dependency order.

        Any arguments to this function will be passed to the callback.
        """
        self.tail_cbs.append((cb, args, kwargs))

    def add_param(self, ptype, name):
        """
        Adds a parameter to this entry. ``type`` and ``name`` are strings.
        """
        if ptype not in TYPES:
            raise TypeError("Unrecognized PTX type name.")
        self._params[name] = Memory(self, 'param', TYPES[ptype], name)

    def add_ptr_param(self, name, mtype):
        """
        Adds a parameter to this entry which points to a location in global
        memory. The resulting property of ``entry.params`` will be a
        ``PtrParam`` for convenient access.

        ``name`` is the param name, and ``mtype`` is the base type of the
        memory location being pointed to. The actual type of the pointer will
        be chosen based on the amount of addressable memory on the device.
        """
        if mtype not in TYPES:
            raise TypeError("Unrecognized PTX type name.")
        # TODO: add pointer size heuristic
        self._params[name] = PtrParam(self, 'global', TYPES[mtype], name)

    def finalize(self):
        """
        This method runs the tail callbacks and performs any introspection
        necessary prior to emitting PTX.
        """
        assert self.tail_cbs is not None, "Cannot finalize more than once!"
        for cb, args, kwargs in reversed(self.tail_cbs):
            cb(*args, **kwargs)
        self.tail_cbs = None

        # This loop verifies rebinding of floating registers to named ones.
        # If all of the conditions below are met, the src register's name will
        # be allowed to match the dst register; otherwise, the src's value
        # will be copied to the dst's with a ``mov`` instruction
        for idx, stmt in enumerate(self._stmts):
            if not isinstance(stmt, Rebinding): continue
            dst, src = stmt
            # src must be floating reg, not immediate or bound reg
            # Examples:
            #   r.a = r.u32(4)
            #   b = r.u32(r.a)
            move = isinstance(src, Immediate) or src.name is not None
            # dst cannot be used between src's originating expression and
            # the rebinding itself
            # Example 1:
            #   r.a, r.b = r.u32(1), r.u32(1)
            #   x = o.add(r.a, r.b)
            #   r.b = o.add(r.a, x)
            #   r.a = x
            # Example 2:
            #   r.a, r.b = r.u32(1), r.u32(1)
            #   label('start')
            #   x = o.add(r.a, r.b)
            #   y = o.add(r.a, x)
            #   r.a = x
            #   r.b = y
            #   bra.uni('start')
            # TODO: incorporate branch tracking
            if not move:
                for oidx in (self._regs[dst].src + self._regs[dst].dst):
                    if oidx > self._regs[src].dst[0] and oidx < idx:
                        move = True
            if move:
                src.rebound_to = None
                stmt = Statement(('mov',), (src,))
                stmt.result = dst
                self._stmts[idx] = stmt

        # Identify all uses of registers by name in the program
        bound = dict([(t, set()) for t in TYPES.values()])
        free = dict([(t, {}) for t in TYPES.values()])
        for stmt in self._stmts:
            if isinstance(stmt, Rebinding):
                regs = [stmt.src, stmt.dst]
            else:
                regs = filter(lambda r: r and not isinstance(r, Immediate),
                              (stmt.result,) + stmt.operands)
            for reg in regs:
                if reg.name:
                    bound[reg.type].add(reg.name)
                else:
                    rl = free[reg.type].setdefault(reg.inferred_name, [])
                    if reg not in rl:
                        rl.append(reg)

        # Store the data required for register declarations
        self.bound = bound
        self.temporary = {}

        # Generate names for all unbound registers
        # TODO: include memory, label, instr identifiers in this list
        identifiers = set()
        map(identifiers.update, bound.values())
        used_bases = set([i.rstrip('1234567890') for i in identifiers])
        for t, inames in free.items():
            for ibase, regs in inames.items():
                if ibase is None:
                    ibase = t.name + '_'
                while ibase in used_bases:
                    ibase = ibase + '_'
                trl = self.temporary.setdefault(t, [])
                trl.append('%s<%d>' % (ibase, len(regs)))
                for i, reg in enumerate(regs):
                    reg.name = ibase + str(i)

    def format_source(self, formatter):
        assert self.tail_cbs is None, "Must finalize entry before formatting"
        params = [v for k, v in sorted(self._params.items())]
        formatter.entry_start(self.name, params, reqntid=self.block)
        [formatter.regs(t, r) for t, r in sorted(self.bound.items()) if r]
        formatter.comment("Temporary registers")
        [formatter.regs(t, r) for t, r in sorted(self.temporary.items()) if r]
        formatter.blank()

        for stmt in self._stmts:
            if isinstance(stmt, Statement):
                stmt.ptx_line = formatter.stmt(stmt)
        formatter.entry_end()


class PTXFormatter(object):
    def __init__(self, ptxver=PTX_VERSION, target='sm_21'):
        self.indent_level = 0
        self.lines = ['.version %d.%d' % ptxver, '.target %s' % target]

    def blank(self):
        self.lines.append('')

    def comment(self, text):
        self.lines.append(' ' * self.indent_level + '// ' + text)

    def regs(self, type, names):
        # TODO: indenting, length limits, etc.
        self.lines.append(' ' * self.indent_level + '.reg .%s ' % (type.name) +
                          ', '.join(sorted(names)) + ';')

    def stmt(self, stmt):
        res = ('%s, ' % stmt.result.get_name()) if stmt.result else ''
        args = [o.get_name() for o in stmt.operands]
        # Wrap the arg in brackets if needed (no good place to put this)
        if stmt.fullname[0] in ('ld ldu st prefetch prefetchu isspacep '
                                'atom red'.split()):
            args[0] = '[%s]' % args[0]

        self.lines.append(''.join([' ' * self.indent_level,
            '%-12s ' % '.'.join(stmt.fullname), res, ', '.join(args), ';']))
        return len(self.lines)

    def entry_start(self, name, params, **directives):
        """
        Define the start of an entry point. ``name`` and ``params`` should be
        obvious, ``directives`` is a dictionary of performance tuning directive
        strings. As a special case, if a ``directive`` value is a tuple, it
        will be converted to a comma-separated string.
        """
        for k, v in directives.items():
            if isinstance(v, tuple):
                directives[k] = ','.join(map(str, v))
        dstr = ' '.join(['.%s %s' % i for i in directives.items()])
        # TODO: support full param options like alignment and array decls
        # (base the param type off a memory type)
        pstrs = ['.param .%s %s' % (p.type.name, p.name) for p in params]
        pstr = '(%s)' % ', '.join(pstrs)
        self.lines.append(' '.join(['.entry', name, pstr, dstr]))
        self.lines.append('{')
        self.indent_level += 4

    def entry_end(self):
        self.indent_level += 4
        self.lines.append('}')

    def get_source(self):
        return '\n'.join(self.lines)

_TExp = namedtuple('_TExp', 'type exprlist')
_DataCell = namedtuple('_DataCell', 'offset size texp')

class DataStream(object):
    """
    Simple interface between Python and PTX, designed to create and tightly
    pack control structs.

    (In the original implementation, this actually used a stack with
    variable positions determined at runtime. The resulting structure had to be
    read strictly sequentially to be parsed, hence the name "stream".)

    Subclass this and give it a shortname, then depend on the subclass in your
    PTX fragments. An instance-based approach is under consideration.

    >>> class ExampleDataStream(DataStream):
    >>>     shortname = "ex"

    Inside DSL functions, you can retrieve arbitrary Python expressions from
    the data stream.

    >>> def example_func():
    >>>     reg.u32('reg1 reg2 regA')
    >>>     op.mov.u32(regA, some_device_allocation_base_address)
    >>>     # From the structure at the base address in 'regA', load the value
    >>>     # of 'ctx.nthreads' into reg1
    >>>     ex.get(regA, reg1, 'ctx.nthreads+padding')

    The expressions will be stored as strings and mapped to particular
    positions in the struct. Later, the expressions will be evaluated and
    coerced into a type matching the destination register:

    >>> data = ExampleDataStream.pack(ctx, padding=4)

    Expressions will be aligned and may be reused in such a way as to minimize
    access times when taking device caching into account. This also implies
    that the evaluated expressions should not modify any state.

    >>> def example_func_2():
    >>>     reg.u32('reg1 reg2')
    >>>     reg.f32('regf')
    >>>     ex.get(regA, reg1, 'ctx.nthreads * 2')
    >>>     # Same expression, so load comes from same memory location
    >>>     ex.get(regA, reg2, 'ctx.nthreads * 2')
    >>>     # Vector loads are pre-coerced, so you can mix types
    >>>     ex.get_v2(regA, reg1, '4', regf, '5.5')

    You can even do device allocations in the file, using the post-finalized
    variable '[prefix]_stream_size'. It's a DelayVar; simple things like
    multiplying by a number work (as long as the DelayVar comes first), but
    fancy things like multiplying two DelayVars aren't implemented yet.

    >>> class Whatever(PTXFragment):
    >>>     def module_setup(self):
    >>>         mem.global_.u32('ex_streams', ex.stream_size*1000)
    """
    # Must be at least as large as the largest load (.v4.u32 = 16)
    alignment = 16

    def __init__(self):
        self.texp_map = {}
        self.cells = []
        self._size = 0
        self.free = {}
        self.size_delayvars = []
        self.finalized = False

    _types = dict(s8='b', u8='B', s16='h', u16='H', s32='i', u32='I', f32='f',
                  s64='l', u64='L', f64='d')
    def _get_type(self, regs):
        size = int(regs[0].type[1:])
        for reg in regs:
            if reg.type not in self._types:
                raise TypeError("Register %s of type %s not supported" %
                                (reg.name, reg.type))
            if int(reg.type[1:]) != size:
                raise TypeError("Can't vector-load different size regs")
        return size/8, ''.join([self._types.get(r.type) for r in regs])

    def _alloc(self, vsize, texp):
        # A really crappy allocator. May later include optimizations for
        # keeping common variables on the same cache line, etc.
        alloc = vsize
        idx = self.free.get(alloc)
        while idx is None and alloc < self.alignment:
            alloc *= 2
            idx = self.free.get(alloc)
        if idx is None:
            # No aligned free cells, allocate a new `align`-byte free cell
            assert alloc not in self.free
            self.free[alloc] = idx = len(self.cells)
            self.cells.append(_DataCell(self._size, alloc, None))
            self._size += alloc
        # Overwrite the free cell at `idx` with texp
        assert self.cells[idx].texp is None
        offset = self.cells[idx].offset
        self.cells[idx] = _DataCell(offset, vsize, texp)
        self.free.pop(alloc)
        # Now reinsert the fragmented free cells.
        fragments = alloc - vsize
        foffset = offset + vsize
        fsize = 1
        fidx = idx
        while fsize < self.alignment:
            if fragments & fsize:
                assert fsize not in self.free
                fidx += 1
                self.cells.insert(fidx, _DataCell(foffset, fsize, None))
                foffset += fsize
                for k, v in filter(lambda (k, v): v >= fidx, self.free.items()):
                    self.free[k] = v+1
                self.free[fsize] = fidx
            fsize *= 2
        return offset

    def _stream_get_internal(self, areg, dregs, exprs, ifp, ifnotp):
        size, type = self._get_type(dregs)
        vsize = size * len(dregs)
        texp = _TExp(type, tuple(exprs))
        if texp in self.texp_map:
            offset = self.texp_map[texp]
        else:
            offset = self._alloc(vsize, texp)
            self.texp_map[texp] = offset
        opname = ['ldu', 'b%d' % (size*8)]
        if len(dregs) > 1:
            opname.insert(1, 'v%d' % len(dregs))
            dregs = vec(*dregs)
        op._call(opname, dregs, addr(areg, offset), ifp=ifp, ifnotp=ifnotp)

    def get(self, areg, dreg, expr, ifp=None, ifnotp=None):
        self._stream_get_internal(areg, [dreg], [expr], ifp, ifnotp)

    def get_v2(self, areg, dreg1, expr1, dreg2, expr2, ifp=None, ifnotp=None):
        self._stream_get_internal(areg, [dreg1, dreg2], [expr1, expr2],
                                  ifp, ifnotp)

    # The interleaved signature makes calls easier to read
    def get_v4(self, areg, d1, e1, d2, e2, d3, e3, d4, e4,
                     ifp=None, ifnotp=None):
        self._stream_get_internal(areg, [d1, d2, d3, d4], [e1, e2, e3, e4],
                                  ifp, ifnotp)

    @property
    def stream_size(self):
        if self.finalized:
            return self._size
        x = DelayVar("not_yet_determined")
        self.size_delayvars.append(x)
        return x

    def finalize_code(self):
        self.finalized = True
        for dv in self.size_delayvars:
            dv.val = self._size
        print "Finalized stream:"
        self._print_format()

    def pack(self, ctx, _out_file_ = None, **kwargs):
        """
        Evaluates all statements in the context of **kwargs. Take this code,
        presumably inside a PTX func::

        >>> ex.get(regA, reg1, 'sum([x+frob for x in xyz.things])')

        To pack this into a struct, call this method on an instance:

        >>> data = ExampleDataStream.pack(ctx, frob=4, xyz=xyz)

        This evaluates each Python expression from the stream with the provided
        arguments as locals, coerces it to the appropriate type, and returns
        the resulting structure as a string.

        The supplied LaunchContext is added to the namespace as ``ctx`` by
        default. To supress, this, override ``ctx`` in the keyword arguments:

        >>> data = ExampleDataStream.pack(ctx, frob=5, xyz=xyz, ctx=None)
        """
        out = StringIO()
        cls.pack_into(out, kwargs)
        return out.read()

    def pack_into(self, ctx, outfile, **kwargs):
        """
        Like pack(), but write data to a file-like object at the file's current
        offset instead of returning it as a string.

        >>> ex_stream.pack_into(ctx, strio_inst, frob=4, xyz=thing)
        >>> ex_stream.pack_into(ctx, strio_inst, frob=6, xyz=another_thing)
        """
        kwargs.setdefault('ctx', ctx)
        for offset, size, texp in self.cells:
            if texp:
                type = texp.type
                vals = [eval(e, globals(), kwargs) for e in texp.exprlist]
            else:
                type = 'x'*size # Padding bytes
                vals = []
            outfile.write(struct.pack(type, *vals))

    def _print_format(self, ctx=None, stream=None):
        for cell in self.cells:
            if cell.texp is None:
                print '%3d %2d --' % (cell.offset, cell.size)
                continue
            print '%3d %2d %4s %s' % (cell.offset, cell.size, cell.texp.type,
                                      cell.texp.exprlist[0])
            for exp in cell.texp.exprlist[1:]:
                print '%11s %s' % ('', exp)

    def print_record(self, ctx, stream, limit=None):
        for i in range(0, len(stream), self._size):
            for cell in self.cells:
                if cell.texp is None:
                    print '%3d %2d --' % (cell.offset, cell.size)
                    continue
                s = '%3d %2d %4s' % (cell.offset, cell.size, cell.texp.type)
                vals = struct.unpack(cell.texp.type,
                                     stream[cell.offset:cell.offset+cell.size])
                for val, exp in zip(vals, cell.texp.exprlist):
                    print '%11s %-20s %s' % (s, val, exp)
                    s = ''
            print '\n----\n'
            if limit is not None:
                limit -= 1
                if limit <= 0: break

