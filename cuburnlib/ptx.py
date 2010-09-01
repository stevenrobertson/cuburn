"""
PTX DSL, a domain-specific language for NVIDIA's PTX.

The DSL doesn't really provide any benefits over raw PTX in terms of type
safety or error checking. Where it shines is in enabling code reuse,
modularization, and dynamic data structures. In particular, the "data stream"
that controls the iterations and xforms in cuflame's device code are much
easier to maintain using this system.
"""

# If you see 'import inspect', you know you're in for a good time
import inspect
import ctypes
from collections import namedtuple

# Okay, so here's what's going on.
#
# We're using Python to create PTX. If we just use Python to make one giant PTX
# module, there's no real reason of going to the trouble of using Python to
# begin with, as the things that this system is good for - modularization, unit
# testing, automated analysis, and data structure generation and optimization -
# pretty much require splitting code up into manageable units. However,
# splitting things up at the level of PTX will greatly reduce performance, as
# the cost of accessing the stack, spilling registers, and reloading data from
# system memory is unacceptably high even on Fermi GPUs. So we want to split
# code up into functions within Python, but not within the PTX.
#
# The challenge here is variable lifetime. A PTX function might declare a
# register at the top of the main block and use it several times throughout the
# function. In Python, we split that up into multiple functions, one to declare
# the registers at the start of the scope and another to make use of them later
# on. This makes it very easy to reuse a class of related PTX functions in
# different device entry points, do unit tests, and so on.
#
# The scope of the class instance is unrelated to the normal scope of names in
# Python. In fact, a function call frequently declares a register that may be
# needed by the parent function. So where to store the information regarding
# the register that was declared at the top of the file (name, type, etc)?
# Well, once declared, a variable remains in scope in PTX until the closing
# brace of the block (curly-braces segment) it was declared in. The natural
# place to store it would be in a Pythonic representation of the block: a block
# object that implements the context manager.
#
# This works well in terms of tracking object lifetime, but it adds a great
# deal of ugliness to the code. What I originally sought was this::
#
#   def load_zero(dest_reg):
#       op.mov.u32(dest_reg, 0)
#   def init_module():
#       reg.u32('hooray_reg')
#       load_zero(hooray_reg)
#
# But using blocks to track state, it would turn in to this ugliness::
#
#   def load_zero(block, dest_reg):
#       block.op.mov.u32(op.dest_reg, 0)
#   def init_module():
#       with Block() as block:
#           block.regs.hooray_reg = block.reg.u32('hooray_reg')
#           load_zero(block, block.regs.hooray_reg)
#
# Eeugh.
#
# Anyway, never one to use an acceptable solution when an ill-conceived hack
# was available, I poked and prodded until I found a way to attain my ideal.
# In short, a function with a 'ptx_func' decorator will be wrapped in a
# _BlockInjector context manager, which will temporarily add values to the
# function's global dictionary in such a way as to mimic the desired behavior.
# The decorator is kind enough to pop the values when exiting. The examples
# below give a clear picture of how to use it, but now you know why this
# abomination was crafted to begin with.


BlockCtx = namedtuple('BlockCtx', 'locals code injectors')
PTXStmt = namedtuple('PTXStmt', 'prefix op vars semi indent')

class _BlockInjector(object):
    """
    A ContextManager that, upon entering a context, loads some keys into a
    dictionary, and upon leaving it, removes those keys. If any keys are
    already in the destination dictionary with a different value, an exception
    is raised.

    Useful if the destination dictionary is a func's __globals__.
    """
    def __init__(self, to_inject, inject_into):
        self.to_inject, self.inject_into = to_inject, inject_into
        self.injected = set()
        self.dead = True
    def inject(self, kv, v=None):
        """Inject a key-value pair (passed either as a tuple or separately.)"""
        k, v = v and (kv, v) or kv
        if k not in self.to_inject:
            self.to_inject[k] = v
        if self.dead:
            return
        if k in self.inject_into:
            if self.inject_into[k] is not v:
                raise KeyError("Key with different value already in dest")
        else:
            self.inject_into[k] = v
            self.injected.add(k)
    def __enter__(self):
        self.dead = False
        map(self.inject, self.to_inject.items())
    def __exit__(self, exc_type, exc_val, tb):
        for k in self.injected:
            del self.inject_into[k]
        self.dead = True

class _Block(object):
    """
    State-tracker for PTX fragments. You should really look at Block and
    PTXModule instead of here.

    For important reasons, the instance must be bound locally as "_block".
    """
    name = '_block'
    def __init__(self):
        self.outer_ctx = BlockCtx({self.name: self}, [], [])
        self.stack = [self.outer_ctx]
    def push_ctx(self):
        self.stack.append(BlockCtx(dict(self.stack[-1].locals), [], []))
    def pop_ctx(self):
        bs = self.stack.pop()
        self.stack[-1].code.append(bs.code)
    def injector(self, func_globals):
        inj = BlockInjector(self.stack[-1].locals, func_globals)
        self.stack[-1].injectors.append(inj)
        return inj
    def inject(self, name, object):
        if name in self.stack[-1].locals:
            raise KeyError("Duplicate name already exists in this scope.")
        self.stack[-1].locals[name] = object
        [inj.inject(name, object) for inj in self.stack[-1].injectors]
    def code(self, prefix='', op='', vars=[], semi=True, indent=0):
        """
        Append a PTX statement (or thereabouts) to the current block.

        - `prefix`: a string which will not be indented, regardless of the
                    current indent level, for labels and predicates.
        - `op`:     a string, aligned to current indent level.
        - `vars`:   a list of strings, with best-effort alignment.
        - `semi`:   whether to terminate the current line with a semicolon.
        - `indent`: integer adjustment to the current indent level.

        For `prefix`, `op`, and `vars`, a "string" can also mean a sequence of
        objects that can be coerced to strings, which will be joined without
        spacing. To keep things simple, nested lists and tuples will be reduced
        in this manner (but not other iterable types). Coercion will not happen
        until after the entire DSL call tree has been walked. This allows a
        class to submit a mutable type (e.g. the trivial `StrVar`) when first
        walked with an undefined value, then substitute the correct value on
        being finalized.

        Details about alignment are available in the `PTXFormatter` class. And
        yes, the only real difference between `prefix`, `op`, and `vars` is in
        final appearance, but it is in fact quite helpful for debugging.
        """
        self.stack[-1].append(PTXStmt(prefix, op, vars, indent))

class StrVar(object):
    """
    Trivial wrapper to allow deferred variable substitution.
    """
    def __init__(self, val=None):
        self.val = val
    def __str__(self):
        return str(val)

def ptx_func(func):
    """
    Decorator function for code in the DSL. Any function which accesses the DSL
    namespace, including declared device variables and objects such as "reg"
    or "op", should be wrapped with this. See Block for some examples.
    """
    def ptx_eval(*args, **kwargs):
        if self.name not in globals():
            parent = inspect.stack()[-2][0]
            if self.name in parent.f_locals:
                block = parent.f_locals[self.name]
            elif self.name in parent.f_globals:
                block = parent.f_globals[self.name]
            else:
                # Couldn't find the _block instance. Fail cryptically to
                # encourage users to read the source (for now)
                raise SyntaxError("Black magic")
        else:
            block = globals()['block']
        with block.injector(func.func_globals):
            func(*args, **kwargs)
    return ptx_eval

class Block(object):
    """
    Limits the lifetime of variables in both PTX (using curly-braces) and in
    the Python DSL (via black magic). This is semantically useful, but should
    not otherwise affect device code (the lifetime of a register is
    aggressively minimized by the compiler).

    >>> with block('This comment will appear at the top of the block'):
    >>>     reg.u32('same_name')
    >>> with block():
    >>>     reg.u64('same_name') # OK, because 'same_name' went out of scope

    PTX variables declared inside a block will be available in any other
    ptx_func called within that block. Note that this flies in the face of
    normal Python behavior! That's why it's a DSL. (This doesn't apply to
    non-PTX variables.)

    >>> @ptx_func
    >>> def fn1():
    >>>     op.mov.u32(reg1, 0)
    >>>
    >>> @ptx_func
    >>> def fn2():
    >>>     print x
    >>>
    >>> @ptx_func
    >>> def fn3():
    >>>     with block():
    >>>         reg.u32('reg1')
    >>>         x = 4
    >>>         fn1() # OK: DSL magic propagates 'reg1' to fn1's namespace
    >>>         fn2() # FAIL: DSL magic doesn't touch regular variables
    >>>     fn1() # FAIL: 'reg1' went out of scope along with the block

    This constructor is available as 'block' in the DSL namespace.
    """
    def __init__(self, block):
        # `block` is the real _block
        self.block = block
        self.comment = None
    def __call__(self, comment=None)
        self.comment = comment
        return self
    def __enter__(self):
        self.block.push_ctx()
        self.block.code(op='{', indent=4)
    def __exit__(self, exc_type, exc_value, tb):
        self.block.code(op='}', indent=-4)
        self.block.pop_ctx()

class _CallChain(object):
    """Handles the syntax for the operator chaining in PTX, like op.mul.u32."""
    def __init__(self, block):
        self.block = block
        self.__chain = []
    def __call__(self, *args, **kwargs):
        assert(self.__chain)
        self._call(chain, *args, **kwargs)
        self.__chain = []
    def __getattr__(self, name):
        if name == 'global_':
            name = 'global'
        self.chain.append(name)
        # Another great crime against the universe:
        return self

class Reg(object):
    """
    Creates one or more registers. The argument should be a string containing
    one or more register names, separated by whitespace; the registers will be
    injected into the DSL namespace on creation, so you do not need to
    rebind them to the same name before use.

    >>> with block():
    >>>     reg.u32('addend product')
    >>>     op.mov.u32(addend, 0)
    >>>     op.mov.u32(product, 0)
    >>> op.mov.u32(addend, 1) # Fails, block unbinds globals on leaving scope

    This constructor is available as 'reg' in the DSL namespace.
    """
    def __init__(self, type, name):
        self.type, self.name = type, name
    def __str__(self):
        return self.name

class _RegFactory(_CallChain):
    """The actual 'reg' object in the DSL namespace."""
    def _call(self, type, names):
        assert len(type) == 1
        type = type[0]
        names = names.split()
        regs = map(lambda n: Reg(type, n), names)
        self.block.code(op='.reg .' + type, vars=names)
        [self.block.inject(r.name, r) for r in regs]

# Pending resolution of the op(regs, guard=x) debate
#class Pred(object):
    #"""
    #Allows for predicated execution of operations.

    #>>> pred('p_some_test p_another_test')
    #>>> op.setp.eq.u32(p_some_test, reg1, reg2)
    #>>> op.setp.and.eq.u32(p_another_test, reg1, reg2, p_some_test)
    #>>> with p_some_test.is_set():
    #>>>     op.ld.global.u32(reg1, addr(areg))

    #Predication supports nested function calls, and will cover all code
    #generated inside the predicate block:

    #>>> with p_another_test.is_unset():
    #>>>     some_ptxdsl_function(reg2)
    #>>>     op.st.global.u32(addr(areg), reg2)

    #It is a syntax error to declare registers,
    #However, multiple predicate blocks cannot be nested. Doing so is a syntax
    #error.

    #>>> with p_some_test.is_set():
    #>>>     with p_another_test.is_unset():
    #>>>         pass
    #SyntaxError: ...
    #"""
    #def __init__(self, name):
        #self.name = name
    #def is_set(self, isnot=False):



class Op(_CallChain):
    """
    Performs an operation.

    >>> op.mov.u32(address, mwc_rng_test_sums)
    >>> op.mad.lo.u32(address, offset, 8, address)
    >>> op.st.global_.v2.u32(addr(address), vec(mwc_a, mwc_b))

    To make an operation conditional on a predicate, use 'ifp' or 'ifnotp':

    >>> reg.pred('p1')
    >>> op.setp.eq.u32(p1, reg1, reg2)
    >>> op.mul.lo.u32(reg1, reg1, reg2, ifp=p1)
    >>> op.add.u32(reg2, reg1, reg2, ifnotp=p1)

    Note that the global state-space should be written 'global_' to avoid
    conflict with the Python keyword. `addr` and `vec` are defined in Mem.

    This constructor is available as 'op' in DSL blocks.
    """
    def _call(self, op, *args, ifp=None, ifnotp=None):
        pred = ''
        if ifp:
            if ifnotp:
                raise SyntaxError("can't use both, fool")
            pred = ['@', ifp]
        if ifnotp:
            pred = ['@!', ifnotp]
        self.block.append_code(pred, '.'.join(op), map(str, args))

class Mem(object):
    """
    Reserve memory, optionally with an array size attached.

    >>> mem.global_.u32('global_scalar')
    >>> mem.local.u32('context_sized_local_array', ctx.threads*4)
    >>> mem.shared.u32('shared_array', 12)
    >>> mem.const.u32('const_array_of_unknown_length', True)

    Like registers, memory allocations are injected into the global namespace
    for use by any functions inside the scope without extra effort.

    >>> with block('move address into memory'):
    >>>     reg.u32('mem_address')
    >>>     op.mov.u32(mem_address, global_scalar)

    This constructor is available as 'mem' in DSL blocks.
    """
    # Pretty much the same as 'Reg', duplicated only for clarity
    def __init__(self, type, name, array, init):
        self.type, self.name, self.array, self.init = type, name, array, init
    def __str__(self):
        return self.name

    @staticmethod
    def vec(*args):
        """
        Prepare vector arguments to a memory operation.

        >>> op.ld.global.v2.u32(vec(reg1, reg2), addr(areg))
        """
        return ['{', [(a, ', ') for a in args][:-1], '}']

    @staticmethod
    def addr(areg, aoffset=''):
        """
        Prepare an address to a memory operation, optionally specifying offset.

        >>> op.st.global.v2.u32(addr(areg), vec(reg1, reg2))
        >>> op.ld.global.v2.u32(vec(reg1, reg2), addr(areg, 8))
        """
        return ['[', areg, aoffset and '+' or '', aoffset, ']']

class _MemFactory(_CallChain):
    """Actual `mem` object"""
    def _call(self, type, name, array=False, initializer=None):
        assert len(type) == 2
        memobj = Mem(type, name, array)
        self.dsl.inject(name, memobj)
        if array is True:
            array = ['[]']
        elif array:
            array = ['[', array, ']']
        else:
            array = []
        if initializer:
            array += [' = ', initializer]
        self.block.code(op=['.%s.%s ' % type, name, array])

class Label(object):
    """
    Specifies the target for a branch. Scoped in PTX? TODO: test.

    >>> label('infinite_loop')
    >>> op.bra.uni('label')
    """
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

class _LabelFactory(object):
    def __init__(self, block):
        self.block = block
    def __call__(self, name):
        self.block.inject(name, Label(name))

class PTXFragment(object):
    def module_setup(self):
        pass

    def entry_setup(self):
        pass

    def entry_teardown(self):
        pass

    def globals(self):
        pass

    def tests(self):
        pass

    def device_init(self, ctx):
        pass

class PTXFragment(object):
    """
    An object containing PTX DSL functions.

    In cuflame, several different versions of a given function may be
    regenerated in rapid succession

    The final compilation pass is guaranteed to have all "tuned" values fixed
    in their final values for the stream.

    Template code will be processed recursively until all "{{" instances have
    been replaced, using the same namespace each time.

    Note that any method which does not depend on 'ctx' can be replaced with
    an instance of the appropriate return type. So, for example, the 'deps'
    property can be a flat list instead of a function.
    """

    def deps(self):
        """
        Returns a list of PTXFragment types on which this object depends
        for successful compilation. Circular dependencies are forbidden,
        but multi-level dependencies should be fine.
        """
        return [DeviceHelpers]

    def inject(self):
        """
        Returns a dict of items to add to the DSL namespace. The namespace will
        be assembled in dependency order before any ptx_funcs are called.
        """
        return {}

    def module_setup(self):
        """
        PTX function to declare things at module scope. It's a PTX syntax error
        to perform operations at this scope, but we don't yet validate that at
        the Python level. A module will call this function on all fragments in
        dependency order.

        If implemented, this function should use an @ptx_func decorator.
        """
        pass

    def entry_setup(self):
        """
        PTX DSL function which will insert code at the start of an entry, for
        initializing variables and stuff like that.  An entry point will call
        this function on all fragments used in that entry point in dependency
        order.

        If implemented, this function should use an @ptx_func decorator.
        """
        pass

    def entry_teardown(self):
        """
        PTX DSL function which will insert code at the end of an entry, for any
        clean-up that needs to be performed. An entry point will call this
        function on all fragments used in the entry point in *reverse*
        dependency order (i.e. fragments which this fragment depends on will be
        cleaned up after this one).

        If implemented, this function should use an @ptx_func decorator.
        """
        pass

    def tests(self, ctx):
        """
        Returns a list of PTXTest classes which will test this fragment.
        """
        return []

    def set_up(self, ctx):
        """
        Do start-of-stream initialization, such as copying data to the device.
        """
        pass
class PTXModule(object):
    """
    Assembles PTX fragments into a module.
    """

    def __init__(self, entries, inject={}, build_tests=False):
        self._block = b = _Block()
        self.initial_inject = dict(inject)
        self._safeupdate(self.initial_inject, dict(block=Block(b),
            mem=_MemFactory(b), reg=_RegFactory(b), op=Op(b),
            label=_LabelFactory(b), _block=b)
        self.needs_recompilation = True
        self.max_compiles = 10
        while self.needs_recompilation:
            self.assemble(entries, build_tests)
            self.max_compiles -= 1

    def deporder(self, unsorted_instances, instance_map, ctx):
        """
        Do a DFS on PTXFragment dependencies, and return an ordered list of
        instances where no fragment depends on any before it in the list.

        `unsorted_instances` is the list of instances to sort.
        `instance_map` is a dict of types to instances.
        """
        seen = {}
        def rec(inst):
            if inst in seen: return seen[inst]
            deps = filter(lambda d: d is not inst, map(instance_map.get,
                       callable(inst.deps) and inst.deps(self) or inst.deps))
            return seen.setdefault(inst, 1+max([0]+map(rec, deps)))
        map(rec, unsorted_instances)
        return sorted(unsorted_instances, key=seen.get)

    def _safeupdate(self, dst, src):
        """dst.update(src), but no duplicates allowed"""
        non_uniq = [k for k in src if k in dst]
        if non_uniq: raise KeyError("Duplicate keys: %s" % ','.join(key))
        dst.update(src)

    def assemble(self, entries, build_tests):
        """
        Build the PTX source for the given set of entries.
        """
        # Get a property, dealing with the callable-or-data thing. This is
        # cumbersome, but flexible; when finished, it may be simplified.
        def pget(prop):
            if callable(prop): return prop(ctx)
            return prop

        instances = {}
        unvisited_entries = list(entries)
        entry_names = {}
        tests = []
        parsed_entries = []
        while unvisited_entries:
            ent = unvisited_entries.pop(0)
            seen, unvisited = set(), [ent]
            while unvisited:
                frag = unvisited.pop(0)
                seen.add(frag)
                inst = instances.setdefault(frag, frag())
                for dep in pget(inst.deps):
                    if dep not in seen:
                        unvisited.append(dep)
                if build_tests:
                    for test in pget(inst.tests):
                        if test not in tests:
                            if test not in instances:
                                unvisited_entries.append(test)
                            tests.append(test)

            tmpl_namespace = {'ctx': ctx}
            entry_start, entry_end = [], []
            for inst in self.deporder(map(instances.get, seen), instances, ctx):
                self._safeupdate(tmpl_namespace, pget(inst.subs))
                entry_start.append(pget(inst.entry_start))
                entry_end.append(pget(inst.entry_end))
            entry_start_tmpl = '\n'.join(filter(None, entry_start))
            entry_end_tmpl = '\n'.join(filter(None, reversed(entry_end)))
            name, args, body = pget(instances[ent].entry)
            tmpl_namespace.update({'_entry_name_': name, '_entry_args_': args,
                '_entry_body_': body, '_entry_start_': entry_start_tmpl,
                '_entry_end_': entry_end_tmpl})

            entry_tmpl = (".entry {{ _entry_name_ }} ({{ _entry_args_ }})\n"
                "{\n{{_entry_start_}}\n{{_entry_body_}}\n{{_entry_end_}}\n}\n")
            parsed_entries.append(multisub(entry_tmpl, tmpl_namespace))
            entry_names[ent] = name

        prelude = []
        tmpl_namespace = {'ctx': ctx}
        for inst in self.deporder(instances.values(), instances, ctx):
            prelude.append(pget(inst.prelude))
            self._safeupdate(tmpl_namespace, pget(inst.subs))
        tmpl_namespace['_prelude_'] = '\n'.join(filter(None, prelude))
        tmpl_namespace['_entries_'] = '\n\n'.join(parsed_entries)
        tmpl = "{{ _prelude_ }}\n{{ _entries_ }}"

        self.entry_names = entry_names
        self.source = ppr_ptx(multisub(tmpl, tmpl_namespace))
        self.instances = instances
        self.tests = tests



class PTXEntryPoint(PTXFragment):
    # Human-readable entry point name
    name = ""

    def entry(self, ctx):
        """
        Returns a 3-tuple of (name, args, body), which will be assembled into
        a function.
        """
        raise NotImplementedError

    def call(self, ctx):
        """
        Calls the entry point on the device. Haven't worked out the details
        of this one yet.
        """
        pass

class PTXTest(PTXEntryPoint):
    """PTXTests are semantically equivalent to PTXEntryPoints, but they
    differ slightly in use. In particular:

    * The "name" property should describe the test being performed,
    * ctx.stream will be synchronized before 'call' is run, and should be
      synchronized afterwards (i.e. sync it yourself or don't use it),
    * call() should return True to indicate that a test passed, or
      False (or raise an exception) if it failed.
    """
    pass

class DeviceHelpers(PTXFragment):
    def __init__(self):
        self._forstack = []

    prelude = ".version 2.1\n.target sm_20\n\n"

    def _get_gtid(self, dst):
        return "{\n// Load GTID into " + dst + """
        .reg .u16 tmp;
        .reg .u32 cta, ncta, tid, gtid;

        mov.u16         tmp,    %ctaid.x;
        cvt.u32.u16     cta,    tmp;
        mov.u16         tmp,    %ntid.x;
        cvt.u32.u16     ncta,   tmp;
        mul.lo.u32      gtid,   cta,    ncta;

        mov.u16         tmp,    %tid.x;
        cvt.u32.u16     tid,    tmp;
        add.u32         gtid,   gtid,   tid;
        mov.b32 """ + dst + ",  gtid;\n}"

    def subs(self, ctx):
        return {
            'PTRT': ctypes.sizeof(ctypes.c_void_p) == 8 and '.u64' or '.u32',
            'get_gtid': self._get_gtid
            }


