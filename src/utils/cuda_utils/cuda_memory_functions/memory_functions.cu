#include <cuda_runtime.h>
#include <memory_functions.h>
#include <iostream>

// Safe CUDA malloc function
void* safeCudaMalloc(size_t size, int* err) {
    void* d_ptr = NULL;
    *err = cudaMalloc(&d_ptr, size);

    if (*err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString((cudaError_t)(*err)));
        return NULL;  // Return NULL if allocation fails
    }

    return d_ptr;  // Return valid pointer if successful
}

// Safe CUDA free function
void safeCudaFree(void* d_ptr) {
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

// Safe CUDA memcpy function
void safeCudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    int err = cudaMemcpy(dst, src, count, kind);

    if (err != cudaSuccess) {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString((cudaError_t)err));
    }
}