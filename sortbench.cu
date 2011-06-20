#include <cuda.h>


__global__
void prefix_scan_8_0_shmem(unsigned char *keys, int nitems, int *pfxs) {
    __shared__ int sh_pfxs[256];

    if (threadIdx.y < 8)
        sh_pfxs[threadIdx.y * 32 + threadIdx.x] = 0;

    __syncthreads();

    int blksz = blockDim.x * blockDim.y;
    int cap = nitems * (blockIdx.x + 1);

    for (int i = threadIdx.y * 32 + threadIdx.x + nitems * blockIdx.x;
         i < cap; i += blksz) {
        int value = keys[i];
        atomicAdd(sh_pfxs + value, 1);
    }

    __syncthreads();

    if (threadIdx.y < 8) {
        int off = threadIdx.y * 32 + threadIdx.x;
        atomicAdd(pfxs + off, sh_pfxs[off]);
    }
}

__global__
void prefix_scan_8_0_shmem_lessconf(unsigned char *keys, int nitems, int *pfxs) {
    __shared__ int sh_pfxs_banked[256][32];

    for (int i = threadIdx.y; i < 256; i += blockDim.y)
        sh_pfxs_banked[i][threadIdx.x] = 0;

    __syncthreads();

    int blksz = blockDim.x * blockDim.y;
    int cap = nitems * (blockIdx.x + 1);

    for (int i = threadIdx.y * 32 + threadIdx.x + nitems * blockIdx.x;
         i < cap; i += blksz) {
        int value = keys[i];
        atomicAdd(&(sh_pfxs_banked[value][threadIdx.x]), 1);
    }

    __syncthreads();

    for (int i = threadIdx.y; i < 256; i += blockDim.y) {
        for (int j = 16; j > 0; j = j >> 1)
            if (j > threadIdx.x)
                sh_pfxs_banked[i][threadIdx.x] += sh_pfxs_banked[i][j+threadIdx.x];
        __syncthreads();
    }

    if (threadIdx.y < 8) {
        int off = threadIdx.y * 32 + threadIdx.x;
        atomicAdd(pfxs + off, sh_pfxs_banked[off][0]);
    }

}

__global__
void prefix_scan_5_0_popc(unsigned char *keys, int nitems, int *pfxs) {
    __shared__ int sh_pfxs[32];

    if (threadIdx.y == 0) sh_pfxs[threadIdx.x] = 0;

    __syncthreads();

    int blksz = blockDim.x * blockDim.y;
    int cap = nitems * (blockIdx.x + 1);

    int sum = 0;

    for (int i = threadIdx.y * 32 + threadIdx.x + nitems * blockIdx.x;
         i < cap; i += blksz) {

        int value = keys[i];
        int test = __ballot(value & 1);
        if (!(threadIdx.x & 1)) test = ~test;

        int popc_res = __ballot(value & 2);
        if (!(threadIdx.x & 2)) popc_res = ~popc_res;
        test &= popc_res;

        popc_res = __ballot(value & 4);
        if (!(threadIdx.x & 4)) popc_res = ~popc_res;
        test &= popc_res;

        popc_res = __ballot(value & 8);
        if (!(threadIdx.x & 8)) popc_res = ~popc_res;
        test &= popc_res;

        popc_res = __ballot(value & 16);
        if (!(threadIdx.x & 16)) popc_res = ~popc_res;
        test &= popc_res;

        sum += __popc(test);
    }

    atomicAdd(sh_pfxs + threadIdx.x + 0,   sum);
    __syncthreads();

    if (threadIdx.y == 0) {
        int off = threadIdx.x;
        atomicAdd(pfxs + off, sh_pfxs[off]);
    }
}


__global__
void prefix_scan_8_0_popc(unsigned char *keys, int nitems, int *pfxs) {
    __shared__ int sh_pfxs[256];

    if (threadIdx.y < 8)
        sh_pfxs[threadIdx.y * 32 + threadIdx.x] = 0;

    __syncthreads();

    int blksz = blockDim.x * blockDim.y;
    int cap = nitems * (blockIdx.x + 1);

    int sum_000 = 0;
    int sum_001 = 0;
    int sum_010 = 0;
    int sum_011 = 0;
    int sum_100 = 0;
    int sum_101 = 0;
    int sum_110 = 0;
    int sum_111 = 0;

    for (int i = threadIdx.y * 32 + threadIdx.x + nitems * blockIdx.x;
         i < cap; i += blksz) {

        int value = keys[i];
        int test_000 = __ballot(value & 1);
        if (!(threadIdx.x & 1)) test_000 = ~test_000;

        int popc_res = __ballot(value & 2);
        if (!(threadIdx.x & 2)) popc_res = ~popc_res;
        test_000 &= popc_res;

        popc_res = __ballot(value & 4);
        if (!(threadIdx.x & 4)) popc_res = ~popc_res;
        test_000 &= popc_res;

        popc_res = __ballot(value & 8);
        if (!(threadIdx.x & 8)) popc_res = ~popc_res;
        test_000 &= popc_res;

        popc_res = __ballot(value & 16);
        if (!(threadIdx.x & 16)) popc_res = ~popc_res;
        test_000 &= popc_res;

        popc_res = __ballot(value & 32);
        int test_001 = test_000 & popc_res;
        popc_res = ~popc_res;
        test_000 &= popc_res;

        popc_res = __ballot(value & 64);
        int test_010 = test_000 & popc_res;
        int test_011 = test_001 & popc_res;
        popc_res = ~popc_res;
        test_000 &= popc_res;
        test_001 &= popc_res;

        popc_res = __ballot(value & 128);
        int test_100 = test_000 & popc_res;
        int test_101 = test_001 & popc_res;
        int test_110 = test_010 & popc_res;
        int test_111 = test_011 & popc_res;
        popc_res = ~popc_res;
        test_000 &= popc_res;
        test_001 &= popc_res;
        test_010 &= popc_res;
        test_011 &= popc_res;

        sum_000 += __popc(test_000);
        sum_001 += __popc(test_001);
        sum_010 += __popc(test_010);
        sum_011 += __popc(test_011);
        sum_100 += __popc(test_100);
        sum_101 += __popc(test_101);
        sum_110 += __popc(test_110);
        sum_111 += __popc(test_111);
    }

    atomicAdd(sh_pfxs + (threadIdx.x + 0),   sum_000);
    atomicAdd(sh_pfxs + (threadIdx.x + 32),  sum_001);
    atomicAdd(sh_pfxs + (threadIdx.x + 64),  sum_010);
    atomicAdd(sh_pfxs + (threadIdx.x + 96),  sum_011);
    atomicAdd(sh_pfxs + (threadIdx.x + 128), sum_100);
    atomicAdd(sh_pfxs + (threadIdx.x + 160), sum_101);
    atomicAdd(sh_pfxs + (threadIdx.x + 192), sum_110);
    atomicAdd(sh_pfxs + (threadIdx.x + 224), sum_111);

    __syncthreads();

    if (threadIdx.y < 8) {
        int off = threadIdx.y * 32 + threadIdx.x;
        atomicAdd(pfxs + off, sh_pfxs[off]);
    }
}

