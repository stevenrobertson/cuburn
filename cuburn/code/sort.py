
import warnings

import numpy as np
import pycuda.driver as cuda
import pycuda.compiler
import tempita

_CODE = tempita.Template(r"""
#include <cuda.h>
#include <stdio.h>

#define GRP_RDX_FACTOR (GRPSZ / RDXSZ)
#define GRP_BLK_FACTOR (GRPSZ / BLKSZ)
#define GRPSZ {{group_size}}
#define RBITS {{radix_bits}}
#define RDXSZ {{radix_size}}
#define BLKSZ 512

#define get_radix(r, k, l) \
        asm("bfe.u32 %0, %1, %2, {{radix_bits}};" : "=r"(r) : "r"(k), "r"(l))

// This kernel conducts a prefix scan of the 'keys' array. As each radix is
// read, it is immediately added to the corresponding accumulator. The
// resulting value is unique among keys in this block with the same radix. We
// will use this offset later (along with the more typical prefix sums) to
// insert the key into a shared memory array by radix, so that they can be
// written in fewer transactions to global memory (which is important given
// the larger radix sizes used here).
//
// Note that the indices generated here are unique but not necessarily
// monotonic, so using these directly leads to a mildly unstable sort.
__global__
void prefix_scan(
        int *offsets,
        int *pfxs,
        const unsigned int *keys,
        const int lo_bit,
        const int ignore_max
) {
    const int tid = threadIdx.x;
    __shared__ int shr_pfxs[RDXSZ];

{{if radix_size <= 512}}
    if (tid < RDXSZ) shr_pfxs[tid] = 0;
{{else}}
{{for i in range(0, radix_size, 512)}}
    shr_pfxs[tid+{{i}}] = 0;
{{endfor}}
{{endif}}

    __syncthreads();
    int idx = tid + GRPSZ * blockIdx.x;

#pragma unroll
    for (int i = 0; i < GRP_BLK_FACTOR; i++) {
        // TODO: load 2 at once, compute, use a BFI to pack the two offsets
        //       into an int to halve storage / bandwidth
        int key = keys[idx];
        if (!ignore_max || key != -1) {
            int radix;
            get_radix(radix, key, lo_bit);
                offsets[idx] = atomicAdd(shr_pfxs + radix, 1);
        }
        idx += BLKSZ;
    }

    __syncthreads();

{{if radix_size <= 512}}
    if (tid < RDXSZ) pfxs[tid + RDXSZ * blockIdx.x] = shr_pfxs[tid];
{{else}}
{{for i in range(0, radix_size, 512)}}
    pfxs[tid + {{i}} + RDXSZ * blockIdx.x] = shr_pfxs[tid + {{i}}];
{{endfor}}
{{endif}}
}

// Populate 'indices' so that indices[i] contains the largest value 'x' such
// that keys[x] < i. If no value is smaller than i, indices[i] is 0.
__global__
void binary_search(
        int *indices,
        const unsigned int *keys,
        const int prev_lo_bit,
        const int mask,
        const int length
) {
    int tid_full = blockIdx.x * blockDim.x + threadIdx.x;
    int target = tid_full << prev_lo_bit;

    int lo = 0;

    // Length must be a power of two! (Guaranteed by runtime.)
    for (int i = length >> 1; i > 0; i >>= 1) {
        int mid = lo | i;
        if (target > (keys[mid] & mask)) lo = mid;
    }

    indices[tid_full] = lo;
}

// When performing a sort by repeatedly applying smaller sorts, the non-stable
// nature of the prefix scan done above will cause errors in the output. These
// errors only affect the correctness of the sort inside those groups which
// cover a transition in the radix of the previous sort passes. We re-run
// those groups with a more careful algorithm here. This doesn't make the sort
// stable in general, but it's enough to make multi-pass sorts correct.
// (Or it should be, although it seems there's a bug either in this code or in
// my head.)
__global__
void prefix_scan_repair(
        int *offsets,
        int *pfxs,
        const unsigned int *keys,
        const unsigned int *trans_points,
        const int lo_bit,
        const int prev_lo_bit,
        const int mask
) {
    const int tid = threadIdx.x;
    const int blkid = blockIdx.y * gridDim.x + blockIdx.x;
    __shared__ int shr_pfxs[RDXSZ];
    __shared__ int blk_starts[GRP_BLK_FACTOR];
    __shared__ int blk_transitions[GRP_BLK_FACTOR];

    // Never need to repair the start of the array.
    if (blkid == 0) return;

    // Get the start index of the group to repair.
    int grp_start = trans_points[blkid] & ~(GRPSZ - 1);

    // If the largest prev_radix in this block is not equal to than our blkid,
    // it means that another thread block will also attend to the same thread
    // block (in which case we cede to it), or the transition point happens to
    // be on a group boundary. In either case, we should bail.
    //
    // Note that prev_* holds a masked but not shifted value.
    int prev_max = keys[grp_start + (GRPSZ - 1)] & mask;
    if (prev_max != (blkid << prev_lo_bit)) return;

    int prev_incr = 1 << prev_lo_bit;

    // For each block of keys that this thread block will analyze, determine
    // how many transitions occur within that block.
    if (tid < GRP_BLK_FACTOR) {
        int prev_lo = keys[grp_start + tid * BLKSZ] & mask;
        int prev_hi = keys[grp_start + tid * BLKSZ + BLKSZ - 1] & mask;
        blk_starts[tid] = prev_lo;
        blk_transitions[tid] = (prev_hi - prev_lo) >> prev_lo_bit;
    }

{{if radix_size <= 512}}
    if (tid < RDXSZ) shr_pfxs[tid] = 0;
{{else}}
{{for i in range(0, radix_size, 512)}}
    shr_pfxs[tid+{{i}}] = 0;
{{endfor}}
{{endif}}
    __syncthreads();

    int idx = grp_start + tid;
    for (int i = 0; i < GRP_BLK_FACTOR; i++) {
        int key = keys[idx];
        int prev_radix = blk_starts[i];
        int this_prev_radix = key & mask;
        int radix;
        get_radix(radix, key, lo_bit);
        if (this_prev_radix == prev_radix)
            offsets[idx] = atomicAdd(shr_pfxs + radix, 1);
        for (int j = 0; j < blk_transitions[i]; j++) {
            __syncthreads();
            prev_radix += prev_incr;
            if (this_prev_radix == prev_radix)
                offsets[idx] = atomicAdd(shr_pfxs + radix, 1);
        }
        idx += BLKSZ;
    }
}

// Calculate group-local exclusive prefix sums (the number of keys in the
// current group with a strictly smaller radix). Must be launched in a
// (32,1,1) block, regardless of block or radix size.
__global__
void calc_local_pfxs(
        int *locals,
        const int *pfxs
) {
    const int tid = threadIdx.x;
    const int tid5 = tid << 5;
    __shared__ int swap[32*32];

    int base = RDXSZ * 32 * blockIdx.x;

    int value = 0;

    // The contents of 32 group radix counts are loaded in 32-element chunks
    // into shared memory, rotated by 1 unit each group to avoid bank
    // conflicts. Each thread in the warp sums across each group serially,
    // updating the values as it goes, then the results are written coherently
    // to global memory.
    //
    // This leaves the SM underloaded, as this only allows 12 warps per SM. It
    // might be better to halve the chunk size and lose some coalescing
    // efficiency; need to benchmark. It's a relatively cheap step, though.

    for (int j = 0; j < RDXSZ / 32; j++) {
        int jj = j << 5;
        for (int i = 0; i < 32; i++) {
            int base_offset = (i << RBITS) + jj + base + tid;
            int swap_offset = (i << 5) + ((i + tid) & 0x1f);
            swap[swap_offset] = pfxs[base_offset];
        }

#pragma unroll
        for (int i = 0; i < 32; i++) {
            int swap_offset = tid5 + ((i + tid) & 0x1f);
            int tmp = swap[swap_offset];
            swap[swap_offset] = value;
            value += tmp;
        }

        for (int i = 0; i < 32; i++) {
            int base_offset = (i << RBITS) + jj + base + tid;
            int swap_offset = (i << 5) + ((i + tid) & 0x1f);
            locals[base_offset] = swap[swap_offset];
        }
    }
}

// All three prefix_sum functions must be called with a block of (RDXSZ, 1, 1).

// Take the prefix scans generated in the first pass and sum them
// vertically (by radix value), sharded into horizontal groups. Store the
// sums by shard and radix in 'condensed'.
__global__
void prefix_sum_condense(
        int *condensed,
        const int *pfxs,
        const int ngrps,
        const int grpwidth
) {
    const int tid = threadIdx.x;
    int sum = 0;

    int idx = grpwidth * blockIdx.x * RDXSZ + tid;
    int maxidx = min(grpwidth * (blockIdx.x + 1), ngrps) * RDXSZ;

    for (; idx < maxidx; idx += RDXSZ) sum += pfxs[idx];

    condensed[blockIdx.x * RDXSZ + tid] = sum;
}

// Sum the partially-condensed sums completely. Scan the sums horizontally.
// Distribute the scanned sums back to the partially-condensed sums.
__global__
void prefix_sum_inner(
        int *glob_pfxs,
        int *condensed,     // input and output
        const int ncondensed
) {
    const int tid = threadIdx.x;
    int sum = 0;
    int idx = tid;
    __shared__ int sums[RDXSZ];

    for (int i = 0; i < ncondensed; i++) {
        sum += condensed[idx];
        idx += RDXSZ;
    }

    // Yeah, the entire device will be stalled on this horribly ineffecient
    // computation, but it only happens once per sort
    sums[tid] = sum;
    __syncthreads();
    sum = 0;

    // Intentionally exclusive indexing here, fixed below
    for (int i = 0; i < tid; i++) sum += sums[i];
    glob_pfxs[tid] = sum + sums[tid];
    __syncthreads();

    sums[tid] = sum;

    idx = tid;
    for (int i = 0; i < ncondensed; i++) {
        int c = condensed[idx];
        condensed[idx] = sum;
        sum += c;
        idx += RDXSZ;
    }
}

// Distribute the partially-condensed sums back to the uncondensed sums.
__global__
void prefix_sum_distribute(
        int *pfxs,          // input and output
        const int *condensed,
        const int ngrps,
        const int grpwidth
) {
    const int tid = threadIdx.x;
    int sum = condensed[blockIdx.x * RDXSZ + tid];

    int idx = grpwidth * blockIdx.x * RDXSZ + tid;
    int maxidx = min(grpwidth * (blockIdx.x + 1), ngrps) * RDXSZ;

    for (; idx < maxidx; idx += RDXSZ) {
        int p = pfxs[idx];
        pfxs[idx] = sum;
        sum += p;
    }
}

__global__
void radix_sort_direct(
        int *sorted_keys,
        const int *keys,
        const int *offsets,
        const int *pfxs
) {
    const int tid = threadIdx.x;
    const int blk_offset = GRPSZ * blockIdx.x;

    int i = tid;
    for (int j = 0; j < GRP_BLK_FACTOR; j++) {
        int value = keys[i+blk_offset];
        int offset = offsets[i+blk_offset];
        sorted_keys[offset] = value;
        i += BLKSZ;
    }
}

#undef BLKSZ
#define BLKSZ {{group_size / 8}}
__global__
void radix_sort(
        int *sorted_keys,
        const int *keys,
        const int *offsets,
        const int *pfxs,
        const int *locals,
        const int lo_bit,
        const int ignore_max
) {
    const int tid = threadIdx.x;
    const int blk_offset = GRPSZ * blockIdx.x;
    __shared__ int shr_offs[RDXSZ];
    __shared__ int defer[GRPSZ];

    const int pfx_i = RDXSZ * blockIdx.x + tid;
    if (tid < RDXSZ) shr_offs[tid] = locals[pfx_i];
    __syncthreads();

    if (ignore_max)
        for (int i = tid; i < GRPSZ; i += BLKSZ) defer[i] = -1;

    for (int i = tid; i < GRPSZ; i += BLKSZ) {
        int key = keys[i+blk_offset];
        if (ignore_max && key == -1) continue;
        int radix;
        get_radix(radix, key, lo_bit);
        int offset = offsets[i+blk_offset] + shr_offs[radix];
        defer[offset] = key;
    }
    __syncthreads();

    if (tid < RDXSZ) shr_offs[tid] = pfxs[pfx_i] - shr_offs[tid];
    __syncthreads();

    int i = tid;
#pragma unroll
    for (int j = 0; j < GRP_BLK_FACTOR; j++) {
        int key = defer[i];
        if (ignore_max && key == -1) continue;
        int radix;
        get_radix(radix, key, lo_bit);
        int offset = shr_offs[radix] + i;
        sorted_keys[offset] = key;
        i += BLKSZ;
    }
}
""")

class Sorter(object):
    mod = None
    group_size = 8192
    radix_bits = 8

    warn_issued = False

    @classmethod
    def init_mod(cls):
        if cls.__dict__.get('mod') is None:
            cls.radix_size = 1 << cls.radix_bits
            code = _CODE.substitute(group_size=cls.group_size,
                    radix_bits=cls.radix_bits, radix_size=cls.radix_size)
            cubin = pycuda.compiler.compile(code)
            cls.mod = cuda.module_from_buffer(cubin)
            with open('/tmp/sort_kern.cubin', 'wb') as fp:
                fp.write(cubin)
            for name in ['prefix_scan', 'prefix_sum_condense',
                         'prefix_sum_inner', 'prefix_sum_distribute',
                         'binary_search', 'prefix_scan_repair']:
                f = cls.mod.get_function(name)
                setattr(cls, name, f)
                f.set_cache_config(cuda.func_cache.PREFER_L1)
            cls.calc_local_pfxs = cls.mod.get_function('calc_local_pfxs')
            cls.radix_sort = cls.mod.get_function('radix_sort')

    def __init__(self, max_size, offsets=None):
        """
        Create a sorter. The sorter will hold on to internal resources for as
        long as it is alive, including an 'offsets' array of size 4*max_size.
        To share this cost, you may pass in an array of at least this size to
        __init__ (to, for instance, share across different bit-widths in a
        multi-pass sort).
        """
        self.init_mod()
        self.max_size = max_size
        assert max_size % self.group_size == 0
        max_grids = max_size / self.group_size

        if offsets is None:
            self.doffsets = cuda.mem_alloc(self.max_size * 4)
        else:
            self.doffsets = offsets
        self.dpfxs = cuda.mem_alloc(max_grids * self.radix_size * 4)
        self.dlocals = cuda.mem_alloc(max_grids * self.radix_size * 4)

        # There are probably better ways to choose how many condensation
        # groups to launch. TODO: maybe pick one if I care
        self.ncond = 32
        self.dcond = cuda.mem_alloc(self.radix_size * self.ncond * 4)
        self.dglobal = cuda.mem_alloc(self.radix_size * 4)

    def warn(self):
        if not self.warn_issued:
            warnings.warn('You know multi-pass is broken, right?',
                          RuntimeWarning, stacklevel=3)
            self.warn_issued = True

    def sort(self, dst, src, size, lo_bit=0, ignore_max=False,
             prev_lo_bit=None, prev_bits=None, stream=None):
        """
        Sort 'src' by the bits from lo_bit+radix_bits to lo_bit, where 0 is
        the LSB. Store the result in 'dst'.

        If 'ignore_max' is True, any key with the value 0xffffffff will be
        effeciently discarded. The number of valid results in the final array
        can be determined by examining the last item in the device array
        pointed to by this class's 'dglobal' property.

        GIANT ENORMOUS WARNING. The single pass sort is not quite stable, even
        with the multi-pass repair kernel. This means that multi-pass sort is
        bugged. Don't use it unless your application can handle non-monotonic
        values in your "sorted" array.

        To perform a multi-pass sort, pass 'prev_lo_bit' and 'prev_bits',
        indicating the lowest bit considered across the entire sort and the
        number of bits previously sorted. This uses 2^(prev_bits+2) bytes of
        memory and performs about group_size*2^(prev_bits+1) extra operations,
        so it's useful up to three passes but probably not four.
        """

        assert size <= self.max_size and size % self.group_size == 0
        grids = size / self.group_size

        self.prefix_scan(self.doffsets, self.dpfxs, src, np.int32(lo_bit),
                         np.int32(ignore_max), block=(512, 1, 1),
                         grid=(grids, 1), stream=stream)

        # Intentionally ignore prev_bits=0
        if prev_bits:
            self.warn()
            assert not (size & (size - 1)), \
                    'Size must be a power of two, due to my enduring laziness'
            didx = cuda.mem_alloc(4 << prev_bits)
            mask = np.uint32(((1 << prev_bits) - 1) << prev_lo_bit)
            self.binary_search(didx, src, np.int32(prev_lo_bit),
                               mask, np.int32(size),
                               block=(128, 1, 1), grid=((1<<prev_bits)/128,1))

            grid=(1 << min(prev_bits, 15), 1 << max(0, prev_bits-15))
            self.prefix_scan_repair(self.doffsets, self.dpfxs, src, didx,
                    np.int32(lo_bit), np.int32(prev_lo_bit), mask,
                    block=(512, 1, 1), grid=grid, stream=stream)

        self.calc_local_pfxs(self.dlocals, self.dpfxs,
            block=(32, 1, 1), grid=(grids / 32, 1), stream=stream)

        ngrps = np.int32(grids)
        grpwidth = np.int32(np.ceil(float(grids) / self.ncond))

        self.prefix_sum_condense(self.dcond, self.dpfxs, ngrps, grpwidth,
            block=(self.radix_size, 1, 1), grid=(self.ncond, 1), stream=stream)
        self.prefix_sum_inner(self.dglobal, self.dcond, np.int32(self.ncond),
            block=(self.radix_size, 1, 1), grid=(1, 1), stream=stream)
        self.prefix_sum_distribute(self.dpfxs, self.dcond, ngrps, grpwidth,
            block=(self.radix_size, 1, 1), grid=(self.ncond, 1), stream=stream)

        self.radix_sort(dst, src, self.doffsets, self.dpfxs, self.dlocals,
            np.int32(lo_bit), np.int32(ignore_max),
            block=(self.group_size / 8, 1, 1), grid=(grids, 1), stream=stream)

    def multisort(self, scratch_a, scratch_b, src, size, lo_bit=0,
                  rounds=1, stream=None):
        """
        Sort 'src', using scratch buffers 'scratch_a' and 'scratch_b' to hold
        the output of intermediate stages. Return whichever of the scratch
        buffers holds the final sorted data.

        It is okay to pass the same array for 'src' and 'scratch_b'.
        Otherwise, 'src' won't be touched.
        """
        if rounds > 1:
            self.warn()
        for i in range(rounds):
            b = i * self.radix_bits
            self.sort(scratch_a, src, size, lo_bit + b, lo_bit, b, stream)
            scratch_a, scratch_b, src = scratch_b, scratch_a, scratch_a
        return src

    @classmethod
    def test(cls, count, correctness=False):
        keys = np.uint32(np.random.randint(0, 1<<cls.radix_bits, size=count))
        dkeys = cuda.to_device(keys)
        dout_a = cuda.mem_alloc(count * 4)
        dout_b = cuda.mem_alloc(count * 4)

        sorter = cls(count)
        stream = cuda.Stream()

        def test_stub(shift, trials=10, rounds=1):
            # Run once so that evt_a doesn't include initialization time
            sorter.multisort(dout_a, dout_b, dkeys, count, shift,
                             rounds, stream=stream)
            evt_a = cuda.Event().record(stream)
            for i in range(trials):
                buf = sorter.multisort(dout_a, dout_b, dkeys, count, shift,
                             rounds, stream=stream)
            evt_b = cuda.Event().record(stream)
            evt_b.synchronize()
            dur = evt_b.time_since(evt_a) / (rounds * trials)
            print '%6.1f,\t%4.0f,\t%4.0f' % (dur, count / (dur * 1000),
                    count * sorter.radix_bits / (dur * 32 * 1000))

            if shift == 0 and correctness:
                print '\nTesting correctness'
                out = cuda.from_device(buf, (count,), np.uint32)
                sort = np.sort(keys)
                if np.all(out == sort):
                    print 'Correct'
                else:
                    nz = np.nonzero(out != sort)[0]
                    print sorted(set(nz >> 13))
                    for i in nz:
                        print i, out[i-1:i+2], sort[i-1:i+2]
                    assert False, 'Oh no'


        for b in range(cls.radix_bits - 3):
            print '%2d (%2d sig bits),\t' % (cls.radix_bits, cls.radix_bits - b),
            test_stub(b)

        if not correctness:
            for r in range(2,3):
                keys[:] = np.uint32(
                        np.random.randint(0, 1<<(cls.radix_bits*r), count))
                cuda.memcpy_htod(dkeys, keys)
                print '%2d x %d,\t\t\t' % (cls.radix_bits, r),
                test_stub(0, rounds=r)
        print

if __name__ == "__main__":
    import sys
    import pycuda.autoinit

    np.set_printoptions(precision=5, edgeitems=200,
                        linewidth=95, threshold=9000)
    count = 1 << 25

    np.random.seed(42)

    correct = '-c' in sys.argv
    for g in (8192, 4096):
        print '\n\n== GROUP SIZE %d ==,\t  msec,\tMK/s,\tMK/s norm' % g
        Sorter.group_size = g
        for b in [7,8,9,10]:
            if g == 4096 and b == 10: continue
            Sorter.radix_bits = b
            Sorter.test(count, correct)
            del Sorter.mod

