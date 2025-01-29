#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

void* safeCudaMalloc(size_t size, int* err);
void safeCudaFree(void* d_ptr);
void safeCudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);

#endif /* CUDA_UTILS_H */
