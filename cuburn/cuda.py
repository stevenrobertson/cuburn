# These imports are order-sensitive!
import pyglet
import pyglet.gl as gl
gl.get_current_context()

import pycuda.driver as cuda
import pycuda.tools
import pycuda.gl as cudagl
import pycuda.gl.autoinit

import numpy as np

from cuburn.ptx import PTXModule, PTXTest, PTXTestFailure

class LaunchContext(object):
    """
    Context collecting the information needed to create, run, and gather the
    results of a device computation. This may eventually also include an actual
    CUDA context, but for now it just uses the global one.

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
    def __init__(self, entries, block=(1,1,1), grid=(1,1), tests=False):
        self.entry_types = entries
        self.block, self.grid, self.build_tests = block, grid, tests
        self.setup_done = False

    @property
    def threads(self):
        return reduce(lambda a, b: a*b, self.block + self.grid)

    @property
    def ctas(self):
        return self.grid[0] * self.grid[1]

    @property
    def threads_per_cta(self):
        return self.block[0] * self.block[1] * self.block[2]

    @property
    def warps_per_cta(self):
        return self.threads_per_cta / 32

    def compile(self, verbose=False, **kwargs):
        kwargs['ctx'] = self
        self.ptx = PTXModule(self.entry_types, kwargs, self.build_tests)
        try:
            self.mod = cuda.module_from_buffer(self.ptx.source)
        except (cuda.CompileError, cuda.RuntimeError), e:
            print "Aww, dang, compile error. Here's the source:"
            self.ptx.print_source()
            raise e
        if verbose:
            if verbose >= 3:
                self.ptx.print_source()
            for entry in self.ptx.entries:
                func = self.mod.get_function(entry.entry_name)
                print "Compiled %s: used %d regs, %d sm, %d local" % (
                        entry.entry_name, func.num_regs,
                        func.shared_size_bytes, func.local_size_bytes)

    def call_setup(self, entry_inst):
        for inst in self.ptx.entry_deps[type(entry_inst)]:
            inst.call_setup(self)

    def call_teardown(self, entry_inst):
        okay = True
        for inst in reversed(self.ptx.entry_deps[type(entry_inst)]):
            if inst is entry_inst and isinstance(entry_inst, PTXTest):
                try:
                    inst.call_teardown(self)
                except PTXTestFailure, e:
                    print "PTX Test %s failed!" % inst.entry_name, e
                    okay = False
            else:
                inst.call_teardown(self)
        return okay

    def run_tests(self):
        if not self.ptx.tests:
            print "No tests to run."
            return True
        all_okay = True
        for test in self.ptx.tests:
            cuda.Context.synchronize()
            if test.call(self):
                print "Test %s passed." % test.entry_name
            else:
                print "Test %s FAILED." % test.entry_name
            all_okay = False
        return all_okay

