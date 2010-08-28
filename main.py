#!/usr/bin/python
#
# flam3cuda, one of a surprisingly large number of ports of the fractal flame
# algorithm to NVIDIA GPUs.
#
# This one is copyright 2010 Steven Robertson <steven@strobe.cc>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 or later
# as published by the Free Software Foundation.

import os
import sys
import time
import ctypes
import struct

import tempita

# These imports are order-sensitive!
import pyglet
import pyglet.gl as gl
gl.get_current_context()

import pycuda.driver as cuda
import pycuda.gl as cudagl
import pycuda.gl.autoinit
from pycuda.compiler import SourceModule

from multiprocessing import Process, Queue

import numpy as np

from fr0stlib import pyflam3

def ppr_ptx(src):
    # TODO: Add variable realignment
    indent = 0
    out = []
    for line in [l.strip() for l in src.split('\n')]:
        if not line:
            continue
        if len(line.split()) == 1 and line.endswith(':'):
            out.append(line)
            continue
        if '}' in line and '{' not in line:
            indent -= 1
        out.append(' ' * (indent * 4) + line)
        if '{' in line and '}' not in line:
            indent += 1
    return '\n'.join(out)

def multisub(tmpl, subs):
    while '{{' in tmpl:
        tmpl = tempita.Template(tmpl).substitute(subs)
    return tmpl

class CUGenome(pyflam3.Genome):
    def _render(self, frame, trans):
        obuf = (ctypes.c_ubyte * ((3+trans)*self.width*self.height))()
        stats = pyflam3.RenderStats()
        pyflam3.flam3_render(ctypes.byref(frame), obuf,
                             pyflam3.flam3_field_both,
                             trans+3, trans, ctypes.byref(stats))
        return obuf, stats, frame

class LaunchContext(object):
    """
    Context collecting the information needed to create, run, and gather the
    results of a device computation.

    To create the fastest device code across multiple device families, this
    context may decide to iteratively refine the final PTX by regenerating
    and recompiling it several times to optimize certain parameters of the
    launch, such as the distribution of threads throughout the device.
    The properties of this device which are tuned are listed below. Any PTX
    fragments which use this information must emit valid PTX for any state
    given below, but the PTX is only required to actually run with the final,
    fixed values of all tuned parameters below.

        `block`:    3-tuple of (x,y,z); dimensions of each CTA.
        `grid`:     2-tuple of (x,y); dimensions of the grid of CTAs.
        `threads`:  Number of active threads on device as a whole.
        `mod`:      Final compiled module. Unavailable during assembly.

    """
    def __init__(self, block=(1,1,1), grid=(1,1), seed=None, tests=False):
        self.block, self.grid, self.tests = block, grid, tests
        self.stream = cuda.Stream()
        self.rand = np.random.mtrand.RandomState(seed)

    @property
    def threads(self):
        return reduce(lambda a, b: a*b, self.block + self.grid)

    def _deporder(self, unsorted_instances, instance_map):
        # Do a DFS on the mapping of PTXFragment types to instances, returning
        # a list of instances ordered such that nothing depends on anything
        # before it in the list
        seen = {}
        def rec(inst):
            if inst in seen: return seen[inst]
            deps = filter(lambda d: d is not inst, map(instance_map.get,
                       callable(inst.deps) and inst.deps(self) or inst.deps))
            return seen.setdefault(inst, 1+max([0]+map(rec, deps)))
        map(rec, unsorted_instances)
        return sorted(unsorted_instances, key=seen.get)

    def _safeupdate(self, dst, src):
        for key, val in src.items():
            if key in dst:
                raise KeyError("Duplicate key %s" % key)
            dst[key] = val

    def assemble(self, entries):
        # Get a property, dealing with the callable-or-data thing
        def pget(prop):
            if callable(prop): return prop(self)
            return prop

        instances = {}
        entries_unvisited = list(entries)
        tests = set()
        parsed_entries = []
        while entries_unvisited:
            ent = entries_unvisited.pop(0)
            seen, unvisited = set(), [ent]
            while unvisited:
                frag = unvisited.pop(0)
                seen.add(frag)
                inst = instances.setdefault(frag, frag())
                for dep in pget(inst.deps):
                    if dep not in seen:
                        unvisited.append(dep)

            tmpl_namespace = {'ctx': self}
            entry_start, entry_end = [], []
            for inst in self._deporder(map(instances.get, seen), instances):
                self._safeupdate(tmpl_namespace, pget(inst.subs))
                entry_start.append(pget(inst.entry_start))
                entry_end.append(pget(inst.entry_end))
            entry_start_tmpl = '\n'.join(filter(None, entry_start))
            entry_end_tmpl = '\n'.join(filter(None, reversed(entry_end)))
            name, args, body = pget(instances[ent].entry)
            tmpl_namespace.update({'_entry_name_': name, '_entry_args_': args,
                '_entry_body_': body, '_entry_start_': entry_start_tmpl,
                '_entry_end_': entry_end_tmpl})

            entry_tmpl = """
.entry {{ _entry_name_ }} ({{ _entry_args_ }})
{
    {{ _entry_start_ }}

    {{ _entry_body_ }}

    {{ _entry_end_ }}
}
"""
            parsed_entries.append(multisub(entry_tmpl, tmpl_namespace))

        prelude = []
        tmpl_namespace = {'ctx': self}
        for inst in self._deporder(instances.values(), instances):
            prelude.append(pget(inst.prelude))
            self._safeupdate(tmpl_namespace, pget(inst.subs))
        tmpl_namespace['_prelude_'] = '\n'.join(filter(None, prelude))
        tmpl_namespace['_entries_'] = '\n\n'.join(parsed_entries)
        tmpl = "{{ _prelude_ }}\n\n{{ _entries_ }}\n"
        return instances, multisub(tmpl, tmpl_namespace)

    def compile(self, entries):
        # For now, do no optimization.
        self.instances, self.src = self.assemble(entries)
        self.src = ppr_ptx(self.src)
        try:
            self.mod = cuda.module_from_buffer(self.src)
        except (cuda.CompileError, cuda.RuntimeError), e:
            print "Aww, dang, compile error. Here's the source:"
            print '\n'.join(["%03d %s" % (i+1, l)
                             for (i, l) in enumerate(self.src.split('\n'))])
            raise e

class PTXFragment(object):
    """
    Wrapper for sections of template PTX.

    In order to provide the best optimization, and avoid a web of hard-coded
    parameters, the PTX module may be regenerated and recompiled several times
    with different or incomplete launch context parameters. To this end, avoid
    accessing the GPU in such functions, and do not depend on context values
    which are marked as "tuned" in the LaunchContext docstring being available.

    The final compilation pass is guaranteed to have all "tuned" values fixed
    in their final values for the stream.

    Template code will be processed recursively until all "{{" instances have
    been replaced, using the same namespace each time.

    Note that any method which does not depend on 'ctx' can be replaced with
    an instance of the appropriate return type. So, for example, the 'deps'
    property can be a flat list instead of a function.
    """

    def deps(self, ctx):
        """
        Returns a list of PTXFragment objects on which this object depends
        for successful compilation. Circular dependencies are forbidden,
        but multi-level dependencies should be fine.
        """
        return [DeviceHelpers]

    def subs(self, ctx):
        """
        Returns a dict of items to add to the template substitution namespace.
        The entire dict will be assembled, including all dependencies, before
        any templates are evaluated.
        """
        return {}

    def prelude(self, ctx):
        """
        Returns a template string containing any code (variable declarations,
        probably) that should be inserted at module scope. The prelude of
        all deps will be inserted above this prelude.
        """
        return ""

    def entry_start(self, ctx):
        """
        Returns a template string that should be inserted at the top of any
        entry point which depends on this method. The entry starts of all
        deps will be inserted above this entry prelude.
        """
        return ""

    def entry_end(self, ctx):
        """
        As above, but at the end of the calling function, and with the order
        reversed (all dependencies will be inserted after this).
        """
        return ""

    def set_up(self, ctx):
        """
        Do start-of-stream initialization, such as copying data to the device.
        """
        pass

    # A list of PTXTest classes which will test this fragment
    tests = []

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

class MWCRNG(PTXFragment):
    def __init__(self):
        if not os.path.isfile('primes.bin'):
            raise EnvironmentError('primes.bin not found')

    prelude = """
.global .u32 mwc_rng_mults[{{ctx.threads}}];
.global .u64 mwc_rng_state[{{ctx.threads}}];"""

    def _next_b32(self, dreg):
        # TODO: make sure PTX optimizes away superfluous move instrs
        return """
    {
    // MWC next b32
    .reg .u64       mwc_out;
    cvt.u64.u32     mwc_out,    mwc_car;
    mad.wide.u32    mwc_out,    mwc_st,     mwc_mult,   mwc_out;
    mov.b64         {mwc_st,    mwc_car},   mwc_out;
    mov.u32         %s,         mwc_st;
    }
    """ % dreg

    def subs(self, ctx):
        return {'mwc_next_b32': self._next_b32}

    entry_start = """
    .reg .u32 mwc_st, mwc_mult, mwc_car;
    {
        // MWC load multipliers and RNG states
        .reg .u32       mwc_off, mwc_addr;
        {{ get_gtid('mwc_off') }}
        mov.u32         mwc_addr,   mwc_rng_mults;
        mad.lo.u32      mwc_addr,   mwc_off,    4,  mwc_addr;
        ld.global.u32   mwc_mult,   [mwc_addr];
        mov.u32         mwc_addr,   mwc_rng_state;
        mad.lo.u32      mwc_addr,   mwc_off,    8,  mwc_addr;
        ld.global.v2.u32 {mwc_st, mwc_car}, [mwc_addr];
    }
    """

    entry_end = """
    {
        // MWC save states
        .reg .u32       mwc_addr, mwc_off;
        {{ get_gtid('mwc_off') }}
        mov.u32         mwc_addr,   mwc_rng_state;
        mad.lo.u32      mwc_addr,   mwc_off,    8,      mwc_addr;
        st.global.v2.u32    [mwc_addr],     {mwc_st, mwc_car};
    }
    """

    def set_up(self, ctx):
        # Load raw big-endian u32 multipliers from primes.bin.
        with open('primes.bin') as primefp:
            dt = np.dtype(np.uint32).newbyteorder('B')
            mults = np.frombuffer(primefp.read(), dtype=dt)
        # Randomness in choosing multipliers is good, but larger multipliers
        # have longer periods, which is also good. This is a compromise.
        # TODO: fix mutability, enable shuffle here
        #ctx.rand.shuffle(mults[:ctx.threads*4])
        # Copy multipliers and seeds to the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        # TODO: get async to work
        #cuda.memcpy_htod_async(multdp, mults.tostring()[:multl], ctx.stream)
        cuda.memcpy_htod(multdp, mults.tostring()[:multl])
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        #cuda.memcpy_htod_async(statedp, ctx.rand.bytes(statel), ctx.stream)
        cuda.memcpy_htod(statedp, ctx.rand.bytes(statel))

    def tests(self, ctx):
        return [MWCRNGTest]

class MWCRNGTest(PTXTest):
    name = "MWC RNG sum-of-threads test"
    deps = [MWCRNG]
    rounds = 200

    prelude = ".global .u64 mwc_rng_test_sums[{{ctx.threads}}];"

    def entry(self, ctx):
        return ('MWC_RNG_test', '', """
    .reg .u64   sum, addl;
    .reg .u32   addend;
    mov.u64     sum,    0;

    {{for round in range(%d)}}
        {{ mwc_next_b32('addend') }}
        cvt.u64.u32     addl,   addend;
        add.u64         sum,    sum,    addl;
    {{endfor}}

    {
        .reg .u32       addr, offset;
        {{ get_gtid('offset') }}
        mov.u32         addr,   mwc_rng_test_sums;
        mad.lo.u32      addr,   offset,     8,      addr;
        st.global.u64   [addr], sum;
    }
    """ % self.rounds)

    def call(self, ctx):
        # Get current multipliers and seeds from the device
        multdp, multl = ctx.mod.get_global('mwc_rng_mults')
        mults = cuda.from_device(multdp, ctx.threads, np.uint32)
        statedp, statel = ctx.mod.get_global('mwc_rng_state')
        fullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        sums = np.zeros(ctx.threads, np.uint64)

        print "Running states forward %d rounds on CPU" % self.rounds
        ctime = time.time()
        for i in range(self.rounds):
            states = fullstates & 0xffffffff
            carries = fullstates >> 32
            fullstates = mults * states + carries
            sums = sums + (fullstates & 0xffffffff)
        ctime = time.time() - ctime
        print "Done on host, took %g seconds" % ctime

        print "Same thing on the device"
        func = ctx.mod.get_function('MWC_RNG_test')
        dtime = func(block=ctx.block, grid=ctx.grid, time_kernel=True)
        print "Done on device, took %g seconds" % dtime

        print "Comparing states and sums..."

        dfullstates = cuda.from_device(statedp, ctx.threads, np.uint64)
        if not (dfullstates == fullstates).all():
            print "State discrepancy"
            print dfullstates
            print fullstates
            #return False

        sumdp, suml = ctx.mod.get_global('mwc_rng_test_sums')
        dsums = cuda.from_device(sumdp, ctx.threads, np.uint64)
        if not (dsums == sums).all():
            print "Sum discrepancy"
            print dsums
            print sums
            return False
        return True


def main(genome_path):
    ctx = LaunchContext(block=(256,1,1), grid=(64,1))
    ctx.compile([MWCRNGTest])
    ctx.instances[MWCRNG].set_up(ctx)
    ctx.instances[MWCRNGTest].call(ctx)



    #with open(genome_path) as fp:
        #genome = CUGenome.from_string(fp.read())[0]
    #genome.width, genome.height = 512, 512
    #genome.sample_density = 1000
    #obuf, stats, frame = genome.render(estimator=3)
    #gc.collect()

        ##q.put(str(obuf))
    ##p = Process(target=render, args=(q, genome_path))
    ##p.start()

    #window = pyglet.window.Window()
    #image = pyglet.image.ImageData(genome.width, genome.height, 'RGB', obuf)
    #tex = image.texture

    #@window.event
    #def on_draw():
        #window.clear()
        #tex.blit(0, 0)

    #pyglet.app.run()

if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print "First argument must be a path to a genome file"
        sys.exit(1)
    main(sys.argv[1])

