#ifndef MEMORY_FUNCTIONS_H
#define MEMORY_FUNCTIONS_H

#include <cuda_runtime.h>

void* safeCudaMalloc(size_t size, int* err);
void safeCudaFree(void* d_ptr);
void safeCudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);

#endif /* MEMORY_FUNCTIONS_H */
