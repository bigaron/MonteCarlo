#include "glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

#include <cuda_runtime.h>
#include <cudaSrc/helper_cuda.h>
#include <cuda_gl_interop.h>

// For threadIdx, blockDim, blockIdx
#include <device_launch_parameters.h>


__global__ void monteCarloKernel(float4* h_odata, int imgw) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y;
	float val = ix / (float)imgw;
	h_odata[iy * imgw + ix] = make_float4(val, 1.0f - val, 1.0f - val, 1.0);
}

extern "C" void computeMonteCarlo(float4* h_odata, dim3 grid, dim3 block, int imgW) {
	monteCarloKernel <<< grid, block >> > (h_odata, imgW);
}