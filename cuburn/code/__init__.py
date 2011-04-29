"""
Contains the PTX fragments which will drive the device.
"""

# Basic headers, utility functions, and so on
base = """
#include<cuda.h>
#include<stdint.h>

// TODO: use launch parameter preconfig to eliminate unnecessary parts
__device__
uint32_t gtid() {
    return threadIdx.x + blockDim.x *
            (threadIdx.y + blockDim.y *
                (threadIdx.z + blockDim.z *
                    (blockIdx.x + (gridDim.x * blockIdx.y))));
}
"""

