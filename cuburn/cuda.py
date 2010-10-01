# These imports are order-sensitive!
#import pyglet
#import pyglet.gl as gl
#gl.get_current_context()

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.tools
#import pycuda.gl as cudagl
#import pycuda.gl.autoinit
import pycuda.autoinit

import numpy as np

from cuburn.ptx import PTXFormatter

class Module(object):
    def __init__(self, entries):
        self.entries = entries
        self.source = self.compile(entries)
        self.mod = self.assemble(self.source)

    @staticmethod
    def compile(entries):
        formatter = PTXFormatter()
        for entry in entries:
            entry.format_source(formatter)
        return formatter.get_source()

    def assemble(self, src):
        # TODO: make this a debugging option
        with open('/tmp/cuburn.ptx', 'w') as f: f.write(src)
        try:
            mod = cuda.module_from_buffer(src,
                [(cuda.jit_option.OPTIMIZATION_LEVEL, 0),
                 (cuda.jit_option.TARGET_FROM_CUCONTEXT, 1)])
        except (cuda.CompileError, cuda.RuntimeError), e:
            # TODO: if output not written above, print different message
            # TODO: read assembler output and recover Python source lines
            print "Compile error. Source is at /tmp/cuburn.ptx"
            print e
            raise e
        return mod

class LaunchContext(object):
    def __init__(self, entries, block=(1,1,1), grid=(1,1), tests=False):
        self.entry_types = entries
        self.block, self.grid, self.build_tests = block, grid, tests
        self.setup_done = False
        self.stream = cuda.Stream()

    @property
    def nthreads(self):
        return reduce(lambda a, b: a*b, self.block + self.grid)

    @property
    def nctas(self):
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
        # TODO: make this optional and let user choose path
        if verbose:
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
                    print "\nTest %s FAILED!" % inst.entry_name
                    print "Reason:", e
                    print
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
                print "Test %s passed.\n" % test.entry_name
            else:
                all_okay = False
        return all_okay

    def get_per_thread(self, name, dtype, shaped=False):
        """
        Convenience function to get the contents of the global memory variable
        ``name`` from the device as a numpy array of type ``dtype``, as might
        be stored by _PTXStdLib.store_per_thread. If ``shaped`` is True, the
        array will be 3D, as (cta_no, warp_no, lane_no).
        """
        if shaped:
            shape = (self.nctas, self.warps_per_cta, 32)
        else:
            shape = self.nthreads
        dp, l = self.mod.get_global(name)
        return cuda.from_device(dp, shape, dtype)

