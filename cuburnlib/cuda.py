# These imports are order-sensitive!
import pyglet
import pyglet.gl as gl
gl.get_current_context()

import pycuda.driver as cuda
import pycuda.tools
import pycuda.gl as cudagl
import pycuda.gl.autoinit

import numpy as np

from cuburnlib.ptx import PTXModule

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

    def set_up(self):
        for inst in self.ptx.deporder(self.ptx.instances.values(),
                                      self.ptx.instances):
            inst.device_init(self)

    def run(self):
        if not self.setup_done: self.set_up()

    def run_test(self, test_type):
        if not self.setup_done: self.set_up()
        inst = self.ptx.instances[test_type]
        print "Running test: %s... " % inst.name
        try:
            cuda.Context.synchronize()
            if inst.call(self):
                print "Test %s passed." % inst.name
            else:
                print "Test %s FAILED." % inst.name
        except Exception, e:
            print "Test %s FAILED (exception thrown)." % inst.name
            raise e

    def run_tests(self):
        map(self.run_test, self.ptx.tests)


