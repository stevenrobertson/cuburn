import time

import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda

import numpy as np

import os
os.environ['PATH'] = ('/usr/x86_64-pc-linux-gnu/gcc-bin/4.4.6:'
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
    mod = pycuda.compiler.SourceModule(src, keep=True)
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


    grids = scale * block / 8192
    print grids
    incr_pfxs = np.zeros((grids + 1, 256), dtype=np.int32)
    shortseg_pfxs = np.zeros(256, dtype=np.int32)
    shortseg_sums = np.zeros(256, dtype=np.int32)
    fun = mod.get_function('prefix_scan_8_0_shmem_shortseg')
    fun.set_cache_config(cuda.func_cache.PREFER_L1)
    t = fun(cuda.In(data), cuda.Out(incr_pfxs),
            block=(32, 8, 1), grid=(grids, 1), time_kernel=True)
    print 'shortseg took %g secs.' % t
    print incr_pfxs[0]
    print incr_pfxs[1]

    split = np.zeros((grids, 256), dtype=np.int32)
    fun = mod.get_function('crappy_split')
    fun.set_cache_config(cuda.func_cache.PREFER_L1)
    t = fun(cuda.In(incr_pfxs), cuda.Out(split),
            block=(32, 8, 1), grid=(grids / 256, 1), time_kernel=True)
    print 'crappy_split took %g secs.' % t
    print split

    fun = mod.get_function('prefix_sum')
    fun.set_cache_config(cuda.func_cache.PREFER_L1)
    t = fun(cuda.InOut(incr_pfxs), np.int32(grids * 256),
            cuda.Out(shortseg_pfxs), cuda.Out(shortseg_sums),
            block=(32, 8, 1), grid=(1, 1), time_kernel=True)
    print 'shortseg_sum took %g secs.' % t
    print 'it worked? %s' % (np.all(shortseg_pfxs == popc_pfxs))
    print shortseg_pfxs
    print shortseg_sums
    print incr_pfxs[1] - incr_pfxs[0]

    sorted = np.zeros(scale * block, dtype=np.int32)
    fun = mod.get_function('sort_8')
    fun.set_cache_config(cuda.func_cache.PREFER_L1)
    t = fun(cuda.In(data), cuda.Out(sorted), cuda.In(incr_pfxs),
            block=(32, 8, 1), grid=(grids, 1), time_kernel=True)
    print 'shortseg_sort took %g secs.' % t
    print 'incr0', incr_pfxs[0]
    print sorted[:100]
    print sorted[-100:]

    sorted = np.zeros(scale * block, dtype=np.int32)
    fun = mod.get_function('sort_8_a')
    t = fun(cuda.In(data), cuda.Out(sorted), cuda.In(incr_pfxs), cuda.In(split),
            block=(32, 8, 1), grid=(grids, 1), time_kernel=True)
    print 'shortseg_sort took %g secs.' % t
    print 'incr0', incr_pfxs[0]
    print sorted[:100]
    print sorted[-100:]




def main():
    # shmem is known good; disable the CPU run to get better info from cuprof
    #go(8, 512<<10, True)
    go(1024, 512<<8, False)
    #go(32768, 8192, False)


main()

