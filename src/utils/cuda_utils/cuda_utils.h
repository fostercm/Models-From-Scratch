#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

__global__ void populateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n);
__global__ void transposeMatrixKernel(const float *matrix, float *transposed, const int m, const int n);
void launchPopulateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n);
void launchTransposeMatrixKernel(const float *matrix, float *transposed, const int m, const int n);
void* safeCudaMalloc(size_t size, int* err);
void safeCudaFree(void* d_ptr);
void safeCudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);

#endif /* CUDA_UTILS_H */
