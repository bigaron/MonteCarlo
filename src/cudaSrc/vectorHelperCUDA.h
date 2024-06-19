#pragma once

//Helper functions so the float4 can be used as a vector from linear algebra
//As of CUDA V12.4 these functions havent been implemented yet

#include <cuda_runtime.h>
#include <cudaSrc/helper_cuda.h>
#include <cuda_gl_interop.h>
// For threadIdx, blockDim, blockIdx
#include <device_launch_parameters.h>


__device__ float length_4(const float4 vec) {
	return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
}

__device__ float4 minus_4(const float4 vec1, const float4 vec2) {
	return make_float4(vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z, vec1.w - vec2.w);
}

__device__ float4 add_4(const float4 vec1, const float4 vec2) {
	return make_float4(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z, vec1.w + vec2.w);
}

__device__ float4 div_scalar_4(const float4 vec, const float scalar) {
	return make_float4(vec.x / scalar, vec.y / scalar, vec.z / scalar, vec.w / scalar);
}