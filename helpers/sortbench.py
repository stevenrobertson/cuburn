import time

import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda

import numpy as np
np.set_printoptions(precision=5, edgeitems=20, linewidth=100, threshold=9000)

import sys, os
os.environ['PATH'] = ('/usr/x86_64-pc-linux-gnu/gcc-bin/4.4.6:'
                     + os.environ['PATH'])

i32 = np.int32

with open('sortbench.cu') as f: src = f.read()
mod = pycuda.compiler.SourceModule(src, keep=True, options=[])

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

def rle(a, n=512):
    pos, = np.where(np.diff(a))
    pos = np.concatenate(([0], pos+1, [len(a)]))
    lens = np.diff(pos)
    return [(a[p], p, l) for p, l in zip(pos, lens)[:n]]


def frle(a, n=512):
    return ''.join(['\n\t%4x %6x %6x' % v for v in rle(a, n)])

# Some reference implementations follow for debugging.
def py_convert_offsets(offsets, split, keys, shift):
    grids = len(offsets)
    new_offs = np.empty((grids, 8192), dtype=np.int32)
    for i in range(grids):
        rdxs = (keys[i] >> shift) & 0xff
        o = split[i][rdxs] + offsets[i]
        new_offs[i][o] = np.arange(8192, dtype=np.int32)
    return new_offs

def py_radix_sort_maybe(keys, offsets, pfxs, split, shift):
    grids = len(offsets)
    idxs = np.arange(8192)

    okeys = np.empty(grids*8192, dtype=np.int32)
    okeys.fill(-1)

    for i in range(grids):
        offs = pfxs[i] - split[i]
        lkeys = keys[i][offsets[i]]
        rdxs = (lkeys >> shift) & 0xff
        glob_offsets = offs[rdxs] + idxs
        okeys[glob_offsets] = lkeys
    return okeys

def go_sort(count, stream=None):
    grids = count / 8192

    keys = np.fromstring(np.random.bytes(count*2), dtype=np.uint16)
    #keys = np.arange(count, dtype=np.uint16)
    #np.random.shuffle(keys)
    mkeys = np.reshape(keys, (grids, 8192))
    vals = np.arange(count, dtype=np.uint32)
    dkeys = cuda.to_device(keys)
    dvals = cuda.to_device(vals)
    print 'Done seeding'

    dpfxs = cuda.mem_alloc(grids * 256 * 4)
    doffsets = cuda.mem_alloc(count * 2)
    launch('prefix_scan_8_0', doffsets, dpfxs, dkeys,
            block=(512, 1, 1), grid=(grids, 1), stream=stream, l1=1)

    dsplit = cuda.mem_alloc(grids * 256 * 4)
    launch('better_split', dsplit, dpfxs,
            block=(32, 1, 1), grid=(grids / 32, 1), stream=stream)

    # This stage will be rejiggered along with the split
    launch('prefix_sum', dpfxs, np.int32(grids * 256),
            block=(256, 1, 1), grid=(1, 1), stream=stream, l1=1)

    launch('convert_offsets', doffsets, dsplit, dkeys, i32(0),
            block=(1024, 1, 1), grid=(grids, 1), stream=stream)
    if not stream:
        offsets = cuda.from_device(doffsets, (grids, 8192), np.uint16)
        split = cuda.from_device(dsplit, (grids, 256), np.uint32)
        pfxs = cuda.from_device(dpfxs, (grids, 256), np.uint32)
        tkeys = py_radix_sort_maybe(mkeys, offsets, pfxs, split, 0)
        #print frle(tkeys & 0xff)

    d_skeys = cuda.mem_alloc(count * 2)
    d_svals = cuda.mem_alloc(count * 4)
    if not stream:
        cuda.memset_d32(d_skeys, 0, count/2)
        cuda.memset_d32(d_svals, 0xffffffff, count)
    launch('radix_sort_maybe', d_skeys, d_svals,
            dkeys, dvals, doffsets, dpfxs, dsplit, i32(0),
            block=(1024, 1, 1), grid=(grids, 1), stream=stream, l1=1)

    if not stream:
        skeys = cuda.from_device_like(d_skeys, keys)
        svals = cuda.from_device_like(d_svals, vals)

        # Test integrity of sort (keys and values kept together):
        #   skeys[i] = keys[svals[i]] for all i
        print 'Integrity: ',
        if np.all(svals < len(keys)) and np.all(skeys == keys[svals]):
            print 'pass'
        else:
            print 'FAIL'

    dkeys, d_skeys = d_skeys, dkeys
    dvals, d_svals = d_svals, dvals

    if not stream:
        cuda.memset_d32(d_skeys, 0, count/2)
        cuda.memset_d32(d_svals, 0xffffffff, count)

    launch('prefix_scan_8_8', doffsets, dpfxs, dkeys,
            block=(512, 1, 1), grid=(grids, 1), stream=stream, l1=1)
    launch('better_split', dsplit, dpfxs,
            block=(32, 1, 1), grid=(grids / 32, 1), stream=stream)
    launch('prefix_sum', dpfxs, np.int32(grids * 256),
            block=(256, 1, 1), grid=(1, 1), stream=stream, l1=1)
    if not stream:
        pre_offsets = cuda.from_device(doffsets, (grids, 8192), np.uint16)
    launch('convert_offsets', doffsets, dsplit, dkeys, i32(8),
            block=(1024, 1, 1), grid=(grids, 1), stream=stream)
    if not stream:
        offsets = cuda.from_device(doffsets, (grids, 8192), np.uint16)
        split = cuda.from_device(dsplit, (grids, 256), np.uint32)
        pfxs = cuda.from_device(dpfxs, (grids, 256), np.uint32)
        tkeys = np.reshape(tkeys, (grids, 8192))

        new_offs = py_convert_offsets(pre_offsets, split, tkeys, 8)
        print np.nonzero(new_offs != offsets)
        fkeys = py_radix_sort_maybe(tkeys, new_offs, pfxs, split, 8)
        #print frle(fkeys)

    launch('radix_sort_maybe', d_skeys, d_svals,
            dkeys, dvals, doffsets, dpfxs, dsplit, i32(8),
            block=(1024, 1, 1), grid=(grids, 1), stream=stream, l1=1)

    if not stream:
        #print cuda.from_device(doffsets, (4, 8192), np.uint16)
        #print cuda.from_device(dkeys, (4, 8192), np.uint16)
        #print cuda.from_device(d_skeys, (4, 8192), np.uint16)

        skeys = cuda.from_device_like(d_skeys, keys)
        svals = cuda.from_device_like(d_svals, vals)

        print 'Integrity: ',
        if np.all(svals < len(keys)) and np.all(skeys == keys[svals]):
            print 'pass'
        else:
            print 'FAIL'

        sorted_keys = np.sort(keys)
        # Test that ordering is correct. (Note that we don't need 100%
        # correctness, so this test should be made "soft".)
        print 'Order: ', 'pass' if np.all(skeys == sorted_keys) else 'FAIL'

        #print frle(skeys, 5120)

def go_sort_old(count, stream=None):
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

def main():
    # shmem is known good; disable the CPU run to get better info from cuprof
    #go(8, 512<<10, True)
    #go(1024, 512<<8, False)
    #go(32768, 8192, False)
    stream = cuda.Stream() if '-s' in sys.argv else None
    go_sort(1<<25, stream)
    if stream:
        stream.synchronize()

main()

