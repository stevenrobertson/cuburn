import time

import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda

import numpy as np

import os
os.environ['PATH'] = ('/usr/x86_64-pc-linux-gnu/gcc-bin/4.4.5:'
                     + os.environ['PATH'])

def go(scale, block, test_cpu):
    data = np.fromstring(np.random.bytes(scale*block), dtype=np.uint8)
    print 'Done seeding'

    if test_cpu:
        a = time.time()
        cpu_pfxs = np.array([np.sum(data == v) for v in range(256)])
        b = time.time()
        print cpu_pfxs
        print 'took %g secs on CPU' % (b - a)

    with open('sortbench.cu') as f: src = f.read()
    mod = pycuda.compiler.SourceModule(src)
    fun = mod.get_function('prefix_scan_8_0_shmem')
    shmem_pfxs = np.zeros(256, dtype=np.int32)
    t = fun(cuda.In(data), np.int32(block), cuda.InOut(shmem_pfxs),
            block=(32, 16, 1), grid=(scale, 1), time_kernel=True)
    print 'shmem took %g secs.' % t
    if test_cpu:
        print 'it worked? %s' % (np.all(shmem_pfxs == cpu_pfxs))

    fun = mod.get_function('prefix_scan_8_0_shmem_lessconf')
    shmeml_pfxs = np.zeros(256, dtype=np.int32)
    t = fun(cuda.In(data), np.int32(block), cuda.InOut(shmeml_pfxs),
            block=(32, 32, 1), grid=(scale, 1), time_kernel=True)
    print 'shmeml took %g secs.' % t
    print 'it worked? %s' % (np.all(shmeml_pfxs == shmem_pfxs))

    fun = mod.get_function('prefix_scan_8_0_popc')
    popc_pfxs = np.zeros(256, dtype=np.int32)
    t = fun(cuda.In(data), np.int32(block), cuda.InOut(popc_pfxs),
            block=(32, 16, 1), grid=(scale, 1), time_kernel=True)
    print 'popc took %g secs.' % t
    print 'it worked? %s' % (np.all(shmem_pfxs == popc_pfxs))

    fun = mod.get_function('prefix_scan_5_0_popc')
    popc5_pfxs = np.zeros(32, dtype=np.int32)
    t = fun(cuda.In(data), np.int32(block), cuda.InOut(popc5_pfxs),
            block=(32, 16, 1), grid=(scale, 1), time_kernel=True)
    print 'popc5 took %g secs.' % t
    print popc5_pfxs



def main():
    # shmem is known good; disable the CPU run to get better info from cuprof
    #go(8, 512<<10, True)
    go(1024, 512<<10, False)


main()

