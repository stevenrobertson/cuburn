import time

import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda

import numpy as np

import sys, os
os.environ['PATH'] = ('/usr/x86_64-pc-linux-gnu/gcc-bin/4.4.6:'
                     + os.environ['PATH'])

with open('sortbench.cu') as f: src = f.read()
mod = pycuda.compiler.SourceModule(src, keep=True)

def launch(name, *args, **kwargs):
    fun = mod.get_function(name)
    if kwargs.pop('l1', False):
        fun.set_cache_config(cuda.func_cache.PREFER_L1)
    if not kwargs.get('stream'):
        kwargs['time_kernel'] = True
    print 'launching %s with %sx%s... ' % (name, kwargs['block'],
                                           kwargs['grid']),
    t = fun(*args, **kwargs)
    if t:
        print 'done (%g secs).' % t
    else:
        print 'done.'


def go(scale, block, test_cpu):
    data = np.fromstring(np.random.bytes(scale*block), dtype=np.uint8)
    print 'Done seeding'

    if test_cpu:
        a = time.time()
        cpu_pfxs = np.array([np.sum(data == v) for v in range(256)])
        b = time.time()
        print cpu_pfxs
        print 'took %g secs on CPU' % (b - a)

    shmem_pfxs = np.zeros(256, dtype=np.int32)
    launch('prefix_scan_8_0_shmem',
            cuda.In(data), np.int32(block), cuda.InOut(shmem_pfxs),
            block=(32, 16, 1), grid=(scale, 1), l1=1)
    if test_cpu:
        print 'it worked? %s' % (np.all(shmem_pfxs == cpu_pfxs))

    shmeml_pfxs = np.zeros(256, dtype=np.int32)
    launch('prefix_scan_8_0_shmem_lessconf',
            cuda.In(data), np.int32(block), cuda.InOut(shmeml_pfxs),
            block=(32, 32, 1), grid=(scale, 1), l1=1)
    print 'it worked? %s' % (np.all(shmeml_pfxs == shmem_pfxs))

    popc_pfxs = np.zeros(256, dtype=np.int32)
    launch('prefix_scan_8_0_popc',
            cuda.In(data), np.int32(block), cuda.InOut(popc_pfxs),
            block=(32, 16, 1), grid=(scale, 1), l1=1)

    popc5_pfxs = np.zeros(32, dtype=np.int32)
    launch('prefix_scan_5_0_popc',
            cuda.In(data), np.int32(block), cuda.InOut(popc5_pfxs),
            block=(32, 16, 1), grid=(scale, 1), l1=1)

def rle(a):
    pos, = np.where(np.diff(a))
    lens = np.diff(np.concatenate((pos, [len(a)])))
    return [(a[p], p, l) for p, l in zip(pos, lens)[:5000]]

def go_sort(count, stream=None):
    data = np.fromstring(np.random.bytes(count), dtype=np.uint8)
    ddata = cuda.to_device(data)
    print 'Done seeding'

    grids = count / 8192
    pfxs = np.zeros((grids + 1, 256), dtype=np.int32)
    dpfxs = cuda.to_device(pfxs)

    launch('prefix_scan_8_0_shmem_shortseg', ddata, dpfxs,
            block=(32, 16, 1), grid=(grids, 1), stream=stream, l1=1)

    #dsplit = cuda.to_device(pfxs)
    #launch('crappy_split', dpfxs, dsplit,
            #block=(32, 8, 1), grid=(grids / 256, 1), stream=stream, l1=1)

    dsplit = cuda.mem_alloc(grids * 256 * 4)
    launch('better_split', dsplit, dpfxs,
            block=(32, 1, 1), grid=(grids / 32, 1), stream=stream)
    #if not stream:
        #split = cuda.from_device_like(dsplit, pfxs)
        #split_ = cuda.from_device_like(dsplit_, pfxs)
        #print np.all(split == split_)

    dshortseg_pfxs = cuda.mem_alloc(256 * 4)
    dshortseg_sums = cuda.mem_alloc(256 * 4)
    launch('prefix_sum', dpfxs, np.int32(grids * 256),
            dshortseg_pfxs, dshortseg_sums,
            block=(32, 8, 1), grid=(1, 1), stream=stream, l1=1)

    dsorted = cuda.mem_alloc(count * 4)
    launch('sort_8', ddata, dsorted, dpfxs,
            block=(32, 16, 1), grid=(grids, 1), stream=stream, l1=1)

    launch('sort_8_a', ddata, dsorted, dpfxs, dsplit,
            block=(32, 32, 1), grid=(grids, 1), stream=stream)
    if not stream:
        sorted = cuda.from_device(dsorted, (count,), np.int32)
        f = lambda r: ''.join(['\n\t%3d %4d %4d' % v for v in r])
        sort_stat = f(rle(sorted))
        with open('dev.txt', 'w') as fp: fp.write(sort_stat)

        sorted_np = np.sort(data)
        np_stat = f(rle(sorted_np))
        with open('cpu.txt', 'w') as fp: fp.write(np_stat)

        print 'is_sorted?', np.all(sorted == sorted_np)

    #data = np.fromstring(np.random.bytes(scale*block), dtype=np.uint16)


def main():
    # shmem is known good; disable the CPU run to get better info from cuprof
    #go(8, 512<<10, True)
    #go(1024, 512<<8, False)
    #go(32768, 8192, False)
    stream = cuda.Stream() if '-s' in sys.argv else None
    go_sort(128<<20, stream)
    if stream:
        stream.synchronize()

main()

