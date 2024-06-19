#include "glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "VertexAttrib.h"
#include "cudaSrc/vectorHelperCUDA.h"
#include "MonteCarloParams.h"

#include <cuda_runtime.h>
#include <cudaSrc/helper_cuda.h>
#include <cuda_gl_interop.h>
// For threadIdx, blockDim, blockIdx
#include <device_launch_parameters.h>

__constant__ float PI = 3.1415926535897932384626433832795;
__constant__ float inf = 100000000.f;
__constant__ float maxWalk;
__constant__ float vertexN;
__constant__ float sampleN;
__constant__ float eps;

//==============================================================
// Rand function from Visual Studio
// Actual source: https://www.shadertoy.com/view/WdXfzl
__device__ int cudaRand(int &seeded){ seeded = seeded * 0x343fd + 0x269ec3; return (seeded >> 16) & 32767; }
__device__ float cudaFrand(int &seeded) { return float(cudaRand(seeded)) / 32767.0; }
__device__ int cudaSrand(int2 p, int frame) {
    int n = frame;
    n = (n << 13) ^ n; n = n * (n * n * 15731 + 789221) + 1376312589; // by Hugo Elias
    n += p.y;
    n = (n << 13) ^ n; n = n * (n * n * 15731 + 789221) + 1376312589;
    n += p.x;
    n = (n << 13) ^ n; n = n * (n * n * 15731 + 789221) + 1376312589;
    return n;
}
//==============================================================

__device__ float closestPoint(const float4 x_o, cudaVertexAttrib* boundPointer) {
    float dist = inf;
    //if (boundPointer[t].col.x > 0.90f) return 10;
    for (int i = 0; i < vertexN; ++i) {
        float distFromCurrentWall = length_4(minus_4(x_o, boundPointer[i].pos));
        if (distFromCurrentWall < dist) dist = distFromCurrentWall;
    }
    return dist;
}


__device__ float4 getBoundaryValue(const float4 x_o, cudaVertexAttrib* boundPointer) {
    float dist = inf;
    float4 ret;
    for (unsigned int i = 0; i < vertexN; ++i) {
        float currentDist = length_4(minus_4(x_o, boundPointer[i].pos));
        if (currentDist < dist) {
            dist = currentDist;
            ret = boundPointer[i].col;
        }
    }

    return ret;
}

__device__ float4 monteCarloEstim(float4 currentOrigin, cudaVertexAttrib* boundPointer, int& seed) {
    int it = 0;
    float radius = closestPoint(currentOrigin, boundPointer);
    while (it < maxWalk && radius > eps) {
        it++;
        float th = cudaFrand(seed) * 2.0 * PI;
        currentOrigin = make_float4(currentOrigin.x + radius * cos(th), currentOrigin.y + radius * sin(th), currentOrigin.z, currentOrigin.w);
        radius = closestPoint(currentOrigin, boundPointer);
    }
    if (it == maxWalk) return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    return getBoundaryValue(currentOrigin, boundPointer);
}

__global__ void monteCarloTextureKernel(cudaSurfaceObject_t surface, cudaVertexAttrib* boundPointer, int imgw, int imgh, int iter) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix > imgw || iy > imgh || ix < 0 || iy < 0) return;
    int seeded = cudaSrand(make_int2(ix, iy), iter);

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 realCoord = make_float4(ix, iy, 0.0f, 1.0f);
    for (int i = 0; i < sampleN; ++i) {
        sum = add_4(sum, monteCarloEstim(realCoord, boundPointer, seeded));
    }

    surf2Dwrite(sum, surface, ix * sizeof(float4), iy);
}

__global__ void showDimensionsKernel(cudaSurfaceObject_t surface, int imgw, int imgh) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x, iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix > imgw || iy > imgh || ix < 0 || iy < 0) return;
    float4 blue = make_float4(0.0, 0.0, 1.0, 1.0);
    float4 yellow = make_float4(0.0, 1.0, 0.0, 1.0);
    int len = 32;
    if ((ix / len) % 2 == 0 && (iy / len) % 2 == 0 || (ix / len) % 2 == (iy / len) % 2) surf2Dwrite(blue, surface, ix * sizeof(float4), iy);
    else surf2Dwrite(yellow, surface, ix * sizeof(float4), iy);
}


extern "C" void setupConstants(const MonteCarloParameters & simulationParams) {
    checkCudaErrors(cudaMemcpyToSymbol(eps, &simulationParams.eps, sizeof(simulationParams.eps)));
    checkCudaErrors(cudaMemcpyToSymbol(maxWalk, &simulationParams.maxWalkN, sizeof(simulationParams.maxWalkN)));
    checkCudaErrors(cudaMemcpyToSymbol(sampleN, &simulationParams.sampleN, sizeof(simulationParams.sampleN)));
    checkCudaErrors(cudaMemcpyToSymbol(vertexN, &simulationParams.vertexN, sizeof(simulationParams.vertexN)));
}

extern "C" void computeMonteCarloWithTexture(cudaSurfaceObject_t surface, cudaVertexAttrib * boundPointer, dim3 grid, dim3 block, int imgW, int imgH, int iteration) {
    //showDimensionsKernel <<<grid, block >>> (surface, imgW, imgH);
    monteCarloTextureKernel <<< grid, block >>> (surface, boundPointer, imgW, imgH, iteration);
}