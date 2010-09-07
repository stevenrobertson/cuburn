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
import types
import struct
from cStringIO import StringIO
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
# code up into functions within Python, but not within the PTX source.
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
# later siblings in the call stack. So where to store the information regarding
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
# But using blocks alone to track names, it would turn in to this mess::
#
#   def load_zero(block, dest_reg):
#       block.op.mov.u32(block.op.dest_reg, 0)
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

def _softjoin(args, sep):
    """Intersperses 'sep' between 'args' without coercing to string."""
    return [[arg, sep] for arg in args[:-1]] + list(args[-1:])

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
    def pop(self, keys):
        """Remove keys from a dictionary, as long as we added them."""
        assert not self.dead
        for k in keys:
            if k in self.injected:
                self.inject_into.pop(k)
                self.injected.remove(k)
    def __enter__(self):
        self.dead = False
        map(self.inject, self.to_inject.items())
    def __exit__(self, exc_type, exc_val, tb):
        # Do some real exceptorin'
        if exc_type is not None: return
        for k in self.injected:
            del self.inject_into[k]
        self.dead = True

class _Block(object):
    """
    State-tracker for PTX fragments. You should really look at Block and
    PTXModule instead of here.

    For important reasons, the instance must be bound locally as "_block".
    """
    name = '_block' # For retrieving from parent scope on first call
    def __init__(self):
        self.reset()
    def reset(self):
        self.outer_ctx = BlockCtx({self.name: self}, [], [])
        self.stack = [self.outer_ctx]
    def clean_injectors(self):
        inj = self.stack[-1].injectors
        [inj.remove(i) for i in inj if i.dead]
    def push_ctx(self):
        self.clean_injectors()
        self.stack.append(BlockCtx(dict(self.stack[-1].locals), [], []))
        # The only reason we should have no injectors in the previous block is
        # if we are hitting a new ptx_func entry point or global declaration at
        # PTX module scope, which means the stack only contains the outer
        # context and the current one (i.e. len(stack) == 2)
        if len(self.stack[-2].injectors) == 0:
            assert len(self.stack) == 2, "Empty injector list too early!"
        # Otherwise, the active injector in the previous block is the one for
        # the Python function which is currently creating a new PTX block, and
        # and it needs to be promoted to the current block
        else:
            self.stack[-1].injectors.append(self.stack[-2].injectors.pop())
    def pop_ctx(self):
        self.clean_injectors()
        bs = self.stack.pop()
        # TODO: figure out why this next line is needed
        [bs.injectors.remove(i) for i in bs.injectors if i.dead]
        self.stack[-1].code.extend(bs.code)
        if len(self.stack) == 1:
            # We're on outer_ctx, so all injectors should be gone.
            assert len(bs.injectors) == 0, "Injector/context mismatch"
            return
        # The only injector should be the one added in push_ctx
        assert len(bs.injectors) == 1, "Injector/context mismatch"
        # Find out which keys were injected while in this context
        diff = set(bs.locals.keys()).difference(
               set(self.stack[-1].locals.keys()))
        # Pop keys and move current injector back down to last context
        last_inj = bs.injectors.pop()
        last_inj.pop(diff)
        self.stack[-1].injectors.append(last_inj)
    def injector(self, func_globals):
        inj = _BlockInjector(dict(self.stack[-1].locals), func_globals)
        self.stack[-1].injectors.append(inj)
        return inj
    def inject(self, name, object):
        if name in self.stack[-1].locals:
            if self.stack[-1].locals[name] is not object:
                raise KeyError("'%s' already exists in this scope." % name)
        else:
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
        class to submit a mutable type (e.g. ``DelayVar``) when first
        walked with an undefined value, then substitute the correct value on
        being finalized.

        Details about alignment are available in the `PTXFormatter` class. And
        yes, the only real difference between `prefix`, `op`, and `vars` is in
        final appearance, but it is in fact quite helpful for debugging.
        """
        self.stack[-1].code.append(PTXStmt(prefix, op, vars, semi, indent))

class DelayVar(object):
    """
    Trivial wrapper to allow deferred variable substitution.
    """
    def __init__(self, val=None):
        self.val = val
    def __str__(self):
        return str(self.val)
    def __mul__(self, other):
        # Oh this is truly egregious
        return DelayVarProxy(self, "self.other.val*" + str(other))

class DelayVarProxy(object):
    def __init__(self, other, expr):
        self.other, self.expr = other, expr
    def __str__(self):
        return str(eval(self.expr))

class _PTXFuncWrapper(object):
    """Enables ptx_func"""
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        if _Block.name in globals():
            block = globals()['block']
        else:
            # Find the '_block' from the enclosing scope
            parent = inspect.stack()[2][0]
            if _Block.name in parent.f_locals:
                block = parent.f_locals[_Block.name]
            elif _Block.name in parent.f_globals:
                block = parent.f_globals[_Block.name]
            else:
                # Couldn't find the _block instance. Fail cryptically to
                # encourage users to read the source (for now)
                raise SyntaxError("Black magic")
        # Create a new function with the modified scope and call it. We could
        # do this in __init__, but it would hide any changes to globals from
        # the module's original scope. Still an option if performance sucks.
        newglobals = dict(self.func.func_globals)
        func = types.FunctionType(self.func.func_code, newglobals,
                                  self.func.func_name, self.func.func_defaults,
                                  self.func.func_closure)
        with block.injector(func.func_globals):
            func(*args, **kwargs)

def ptx_func(func):
    """
    Decorator function for code in the DSL. Any function which accesses the DSL
    namespace, including declared device variables and objects such as "reg"
    or "op", should be wrapped with this. See Block for some examples.

    Note that writes to global variables will silently fail for now.
    """
    # Attach most of the code to the wrapper class
    fw = _PTXFuncWrapper(func)
    def wr(*args, **kwargs):
        fw(*args, **kwargs)
    return wr

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
    def __call__(self, comment=None):
        self.comment = comment
        return self
    def __enter__(self):
        self.block.push_ctx()
        self.block.code(op='{', semi=False)
        self.block.code(indent=1)
        if self.comment:
            self.block.code(op=['// ', self.comment], semi=False)
        self.comment = None
    def __exit__(self, exc_type, exc_value, tb):
        # Allow exceptions to be propagated; things get really messy if we try
        # to pop the stack if things aren't ordered correctly
        if exc_type is not None: return
        self.block.code(indent=-1)
        self.block.code(op='}', semi=False)
        self.block.pop_ctx()

class _CallChain(object):
    """Handles the syntax for the operator chaining in PTX, like op.mul.u32."""
    def __init__(self, block):
        self.block = block
        self.__chain = []
    def __call__(self, *args, **kwargs):
        assert(self.__chain)
        r = self._call(self.__chain, *args, **kwargs)
        self.__chain = []
        return r
    def __getattr__(self, name):
        if name.endswith('_'):
            name = name[:-1]
        self.__chain.append(name)
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
        self.block.code(op='.reg .' + type, vars=_softjoin(names, ','))
        [self.block.inject(r.name, r) for r in regs]
        if len(regs) == 1:
            return regs[0]
        return regs

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
    def _call(self, op, *args, **kwargs):
        pred = ''
        ifp = kwargs.get('ifp')
        ifnotp = kwargs.get('ifnotp')
        if ifp:
            if ifnotp:
                raise SyntaxError("can't use both, fool")
            pred = ['@', ifp]
        if ifnotp:
            pred = ['@!', ifnotp]
        self.block.code(pred, '.'.join(op), _softjoin(args, ','))

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
        assert len(args) >= 2, "vector loads make no sense with < 2 args"
        # TODO: fix the way this looks (not so easy)
        return ['{', _softjoin(args, ','), '}']

    @staticmethod
    def addr(areg, aoffset=''):
        """
        Prepare an address to a memory operation, optionally specifying offset.

        >>> op.st.global.v2.u32(addr(areg), vec(reg1, reg2))
        >>> op.ld.global.v2.u32(vec(reg1, reg2), addr(areg, 8))
        """
        return ['[', areg, aoffset is not '' and '+' or '', aoffset, ']']

class _MemFactory(_CallChain):
    """Actual `mem` object"""
    def _call(self, type, name, array=False, initializer=None):
        assert len(type) == 2
        memobj = Mem(type, name, array, initializer)
        if array is True:
            array = ['[]']
        elif array:
            array = ['[', array, ']']
        else:
            array = []
        if initializer:
            array += [' = ', initializer]
        self.block.code(op=['.%s.%s ' % (type[0], type[1]), name, array])
        self.block.inject(name, memobj)

class Label(object):
    """
    Specifies the target for a branch.

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
        self.block.code(prefix='%s:' % name, semi=False)

class Comment(object):
    """Add a single-line comment to the PTX output."""
    def __init__(self, block):
        self.block = block
    def __call__(self, comment):
        self.block.code(op=['// ', comment], semi=False)

class PTXFragment(object):
    """
    An object containing PTX DSL functions. The object, and all its
    dependencies, will be instantiated by a PTX module. Each object will be
    bound to the name given by ``shortname`` in the DSL namespace.
    """

    # Name under which to make this code available in ptx_funcs
    shortname = None

    def deps(self):
        """
        Returns a list of PTXFragment types on which this object depends
        for successful compilation. Circular dependencies are forbidden,
        but multi-level dependencies should be fine.
        """
        return [_PTXStdLib]

    def module_setup(self):
        """
        PTX function to declare things at module scope. It's a PTX syntax error
        to perform operations at this scope, but we don't yet validate that at
        the Python level. A module will call this function on all fragments
        used in that module in dependency order.

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

    def finalize_code(self):
        """
        Called after running all PTX DSL functions, but before code generation,
        to allow fragments which postponed variable evaluation (e.g. using
        ``DelayVar``) to fill in the resulting values. Most fragments should
        not use this.

        If implemented, this function *may* use an @ptx_func decorator to
        access the global DSL scope, but pretty please don't emit any code
        while you're in there.
        """
        pass

    def tests(self):
        """
        Returns a list of PTXTest types which will test this fragment.
        """
        return []

    def device_init(self, ctx):
        """
        Do stuff on the host to prepare the device for execution. 'ctx' is a
        LaunchContext or similar. This will get called (in dependency order, of
        course) *either* before any entry point invocation, or before *each*
        invocation, I'm not sure which yet. (For now it's "each".)
        """
        pass

class PTXEntryPoint(PTXFragment):
    # Human-readable entry point name
    name = ""
    # Device code entry name
    entry_name = ""
    # List of (type, name) pairs for entry params, e.g. [('u32', 'thing')]
    entry_params = []

    def entry(self):
        """
        PTX DSL function that comprises the body of the PTX statement.

        Must be implemented and decorated with ptx_func.
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

class _PTXStdLib(PTXFragment):
    shortname = "std"
    def __init__(self, block):
        # Only module that gets the privilege of seeing 'block' directly.
        self.block = block

    def deps(self):
        return []

    @ptx_func
    def module_setup(self):
        # TODO: make this modular, maybe? of course, we'd have to support
        # multiple devices first, which we definitely do not yet do
        self.block.code(prefix='.version 2.1', semi=False)
        self.block.code(prefix='.target sm_20', semi=False)

    @ptx_func
    def get_gtid(self, dst):
        """
        Get the global thread ID (the position of this thread in a grid of
        blocks of threads). This assumes that both grid and block are
        one-dimensional! (This is always true for cuburn.)
        """
        with block("Load GTID into %s" % str(dst)):
            reg.u32('cta ncta tid')
            op.mov.u32(cta, '%ctaid.x')
            op.mov.u32(ncta, '%ntid.x')
            op.mov.u32(tid, '%tid.x')
            op.mad.lo.u32(dst, cta, ncta, tid)

    @ptx_func
    def store_per_thread(self, base, val):
        """Store b32 at `base+gtid*4`. Super-common debug pattern."""
        with block("Per-thread store of %s" % str(val)):
            reg.u32('spt_base spt_offset')
            op.mov.u32(spt_base, base)
            self.get_gtid(spt_offset)
            op.mad.lo.u32(spt_base, spt_offset, 4, spt_base)
            op.st.b32(addr(spt_base), val)

    @ptx_func
    def set_is_first_thread(self, p_dst):
        with block("Set %s if this is thread 0 in the CTA" % p_dst.name):
            reg.u32('tid')
            op.mov.u32(tid, '%tid.x')
            op.setp.eq.u32(p_dst, tid, 0)

    def not_(self, pred):
        return ['!', pred]

    def to_inject(self):
        # Set up the initial namespace
        return dict(
            _block=self.block,
            block=Block(self.block),
            op=Op(self.block),
            reg=_RegFactory(self.block),
            mem=_MemFactory(self.block),
            addr=Mem.addr,
            vec=Mem.vec,
            label=_LabelFactory(self.block),
            comment=Comment(self.block))

class PTXModule(object):
    """
    Assembles PTX fragments into a module. The following properties are
    available:

    `instances`:    Mapping of type to instance for the PTXFragments used in
                    the creation of this PTXModule.
    `entries`:      List of PTXEntry types in this module, including any tests.
    `tests`:        List of PTXTest types in this module.
    `source`:       PTX source code for this module.
    """
    max_compiles = 10

    def __init__(self, entries, inject={}, build_tests=False, formatter=None):
        """
        Construct a PTXModule.

        `entries`:      List of PTXEntry types to include in this module.
        `inject`:       Dict of items to inject into the DSL namespace.
        `build_tests`:  If true, build tests into the module.
        `formatter`:    PTXFormatter instance, or None to use defaults.
        """
        block = _Block()
        insts, tests, all_deps, entry_deps = (
                self.deptrace(block, entries, build_tests))
        self.instances = insts
        self.tests = tests

        inject = dict(inject)
        inject.update(insts[_PTXStdLib].to_inject())
        self._safeupdate(inject, 'module', self)
        for inst in all_deps:
            if inst.shortname:
                self._safeupdate(inject, inst.shortname, inst)
        [block.inject(k, v) for k, v in inject.items()]

        self.__needs_recompilation = True
        self.compiles = 0
        while self.__needs_recompilation:
            self.compiles += 1
            self.__needs_recompilation = False
            self.assemble(block, all_deps, entry_deps)
        self.instances.pop(_PTXStdLib)

        if not formatter:
            formatter = PTXFormatter()
        self.source = formatter.format(block.outer_ctx.code)
        self.entries = list(set(entries + tests))

    def deporder(self, unsorted_instances, instance_map):
        """
        Do a DFS on PTXFragment dependencies, and return an ordered list of
        instances where no fragment depends on any before it in the list.

        `unsorted_instances` is the list of instances to sort.
        `instance_map` is a dict of types to instances.
        """
        seen = {}
        def rec(inst):
            if inst in seen: return seen[inst]
            if inst is None: return 0
            deps = filter(lambda d: d is not inst,
                          map(instance_map.get, inst.deps()))
            return seen.setdefault(inst, 1+max([0]+map(rec, deps)))
        map(rec, unsorted_instances)
        return sorted(unsorted_instances, key=seen.get)

    def _safeupdate(self, dst, k, v):
        if k in dst: raise KeyError("Duplicate key %s" % k)
        dst[k] = v

    def deptrace(self, block, entries, build_tests):
        instances = {_PTXStdLib: _PTXStdLib(block)}
        unvisited_entries = list(entries)
        tests = set()
        entry_deps = {}

        # For each PTXEntry or PTXTest, use a BFS to recursively find and
        # instantiate all fragments that are dependencies. If tests are
        # discovered, add those to the list of entries.
        while unvisited_entries:
            ent = unvisited_entries.pop(0)
            seen, unvisited = set(), [ent]
            while unvisited:
                frag = unvisited.pop(0)
                seen.add(frag)
                # setdefault doesn't work because of _PTXStdLib
                if frag not in instances:
                    inst = frag()
                    instances[frag] = inst
                else:
                    inst = instances[frag]
                for dep in inst.deps():
                    if dep not in seen:
                        unvisited.append(dep)
                if build_tests:
                    for test in inst.tests():
                        if test not in tests:
                            tests.add(test)
                            if test not in instances:
                                unvisited_entries.append(test)
            # For this entry, store insts of all dependencies in order.
            entry_deps[ent] = self.deporder(map(instances.get, seen),
                                            instances)
        # Find the order for all dependencies in the program.
        all_deps = self.deporder(instances.values(), instances)

        return instances, sorted(tests, key=str), all_deps, entry_deps

    def assemble(self, block, all_deps, entry_deps):
        # Rebind to local namespace to allow proper retrieval
        _block = block

        for inst in all_deps:
            inst.module_setup()

        for ent, insts in entry_deps.items():
            # This is kind of hackish compared to everything else
            params = [Reg('.param.' + str(type), name)
                      for (type, name) in ent.entry_params]
            _block.code(op='.entry %s ' % ent.entry_name, semi=False,
                vars=['(', ', '.join(['%s %s' % (r.type, r.name)
                                      for r in params]), ')'])
            with Block(_block):
                [_block.inject(r.name, r) for r in params]
                for dep in insts:
                    dep.entry_setup()
                self.instances[ent].entry()
                for dep in reversed(insts):
                    dep.entry_teardown()

        for inst in all_deps:
            inst.finalize_code()

    def set_needs_recompilation(self):
        if not self.__needs_recompilation:
            if self.compiles >= self.max_compiles:
                raise ValueError("Too many recompiles scheduled!")
            self.__needs_recompilation = True

    def print_source(self):
        if not hasattr(self, 'source'):
            raise ValueError("Not assembled yet!")
        print '\n'.join(["%03d %s" % (i+1, l) for (i, l) in
                        enumerate(self.source.split('\n'))])


def _flatten(val):
    if isinstance(val, (list, tuple)):
        return ''.join(map(_flatten, val))
    return str(val)

class PTXFormatter(object):
    """
    Formats PTXStmt items into beautiful code. Well, the beautiful part is
    postponed for now.
    """
    def __init__(self, indent_amt=4, oplen_max=20, varlen_max=12):
        self.idamt, self.opm, self.vm = indent_amt, oplen_max, varlen_max
    def format(self, code):
        out = []
        indent = 0
        idx = 0
        while idx < len(code):
            opl, vl = 0, 0
            flat = []
            while idx < len(code):
                pfx, op, vars, semi, indent_change = code[idx]
                idx += 1
                if indent_change: break
                pfx, op = _flatten(pfx), _flatten(op)
                vars = map(_flatten, vars)
                if len(op) <= self.opm:
                    opl = max(opl, len(op)+2)
                for var in vars:
                    if len(var) <= self.vm:
                        vl = max(vl, len(var)+1)
                flat.append((pfx, op, vars, semi))
            for pfx, op, vars, semi in flat:
                if pfx:
                    line = ('%%-%ds ' % (indent-1)) % pfx
                else:
                    line = ' '*indent
                line = ('%%-%ds ' % (indent+opl)) % (line+op)
                for i, var in enumerate(vars):
                    line = ('%%-%ds ' % (indent+opl+vl*(i+1))) % (line+var)
                if semi:
                    line = line.rstrip() + ';'
                out.append(line)
            indent = max(0, indent + self.idamt * indent_change)
        return '\n'.join(out)

_TExp = namedtuple('_TExp', 'type exprlist')
_DataCell = namedtuple('_DataCell', 'offset size texp')

class DataStream(PTXFragment):
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

    Inside DSL functions, you can "retrieve" arbitrary Python expressions from
    the data stream.

    >>> @ptx_func
    >>> def example_func():
    >>>     reg.u32('reg1 reg2 regA')
    >>>     op.mov.u32(regA, some_device_allocation_base_address)
    >>>     # From the structure at the base address in 'regA', load the value
    >>>     # of 'ctx.nthreads' into reg1
    >>>     ex.get(regA, reg1, 'ctx.nthreads')

    The expressions will be stored as strings and mapped to particular
    positions in the struct. Later, the expressions will be evaluated and
    coerced into a type matching the destination register:

    >>> # Fish the instance holding the data stream from the compiled module
    >>> ex_stream = launch_context.ptx.instances[ExampleDataStream]
    >>> # Evaluate the expressions in the current namespace, augmented with the
    >>> # supplied objects
    >>> data = ex_stream.pack(ctx=launch_context)

    Expressions will be aligned and may be reused in such a way as to minimize
    access times when taking device caching into account. This also implies
    that the evaluated expressions should not modify any state, but that should
    be obvious, no?

    >>> @ptx_func
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
    >>>     @ptx_func
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
        # Now reinsert the fragmented free cells.
        fragments = alloc - vsize
        foffset = offset + vsize
        fsize = 1
        fidx = idx
        while fsize <= self.alignment:
            if fragments & fsize:
                assert fsize not in self.free
                fidx += 1
                self.cells.insert(fidx, _DataCell(foffset, fsize, None))
                foffset += fsize
                self.free[fsize] = fidx
            fsize *= 2
        # Adjust indexes. This is ugly, but evidently unavoidable
        if fidx-idx:
            for k, v in filter(lambda (k, v): v > idx, self.free.items()):
                self.free[k] = v+(fidx-idx)
        return offset

    @ptx_func
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
            dregs = vec(dregs)
        op._call(opname, dregs, addr(areg, offset), ifp=ifp, ifnotp=ifnotp)

    @ptx_func
    def get(self, areg, dreg, expr, ifp=None, ifnotp=None):
        self._stream_get_internal(areg, [dreg], [expr], ifp, ifnotp)

    @ptx_func
    def get_v2(self, areg, dreg1, expr1, dreg2, expr2, ifp=None, ifnotp=None):
        self._stream_get_internal(areg, [dreg1, dreg2], [expr1, expr2],
                                  ifp, ifnotp)

    # The interleaved signature makes calls easier to read
    @ptx_func
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

    def pack(self, _out_file_ = None, **kwargs):
        """
        Evaluates all statements in the context of **kwargs. Take this code,
        presumably inside a PTX func::

        >>> ex.get(regA, reg1, 'sum([x+frob for x in xyz.things])')

        To pack this into a struct, call this method on an instance:

        >>> ex_stream = launch_context.ptx.instances[ExampleDataStream]
        >>> data = ex_stream.pack(frob=4, xyz=xyz)

        This evaluates each Python expression from the stream with the provided
        arguments as locals, coerces it to the appropriate type, and returns
        the resulting structure as a string.
        """
        out = StringIO()
        self.pack_into(out, kwargs)
        return out.read()

    def pack_into(self, outfile, **kwargs):
        """
        Like pack(), but write data to a file-like object at the file's current
        offset instead of returning it as a string.

        >>> ex_stream.pack_into(strio_inst, frob=4, xyz=thing)
        >>> ex_stream.pack_into(strio_inst, frob=6, xyz=another_thing)
        """
        for offset, size, texp in self.cells:
            if texp:
                type = texp.type
                vals = [eval(e, globals(), kwargs) for e in texp.exprlist]
            else:
                type = 'x'*size # Padding bytes
                vals = []
            outfile.write(struct.pack(type, *vals))

    def print_record(self):
        for cell in self.cells:
            if cell.texp is None:
                print '%3d %2d --' % (cell.offset, cell.size)
                continue
            print '%3d %2d %4s %s' % (cell.offset, cell.size, cell.texp.type,
                                      cell.texp.exprlist[0])
            for exp in cell.texp.exprlist[1:]:
                print '%12s %s' % ('', exp)

