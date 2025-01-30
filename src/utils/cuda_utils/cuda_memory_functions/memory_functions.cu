/**
 * @file memory_functions.cu
 * @brief CUDA memory management utility functions.
 *
 * This file provides safe CUDA memory management functions for memory allocation, deallocation, and data transfer. 
 * These functions ensure error handling and appropriate cleanup during the memory operations in CUDA-based applications.
 */

#include <cuda_memory_functions/memory_functions.h>
#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief Safely allocates memory on the GPU.
 *
 * This function wraps `cudaMalloc` to handle memory allocation on the GPU, ensuring that errors are checked and handled.
 * If the allocation fails, an error message is printed, and `NULL` is returned. If the allocation is successful, 
 * a valid pointer to the allocated memory is returned.
 *
 * @param[in] size The size (in bytes) of the memory block to allocate on the GPU.
 * @param[out] err Pointer to an integer that will store the CUDA error code, or `cudaSuccess` if allocation succeeds.
 *
 * @return A pointer to the allocated memory on the GPU if successful, or `NULL` if allocation fails.
 */
void* safeCudaMalloc(size_t size, int* err) {
    void* d_ptr = NULL;
    *err = cudaMalloc(&d_ptr, size);

    if (*err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString((cudaError_t)(*err)));
        return NULL;  // Return NULL if allocation fails
    }

    return d_ptr;  // Return valid pointer if successful
}

/**
 * @brief Frees memory on the GPU.
 *
 * This function wraps `cudaFree` to ensure that memory deallocation occurs only if a valid pointer is provided.
 * It is a safeguard to prevent errors caused by freeing memory that has already been deallocated or is `NULL`.
 *
 * @param[in] d_ptr Pointer to the memory to free on the GPU. If the pointer is `NULL`, no action is taken.
 *
 * @note This function does not check for memory allocation errors. It is assumed that the memory was allocated 
 * using a safe CUDA malloc function or is a valid pointer.
 */
void safeCudaFree(void* d_ptr) {
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

/**
 * @brief Safely copies data between host and device memory.
 *
 * This function wraps `cudaMemcpy` to handle the data transfer between host (CPU) and device (GPU) memory. 
 * If the transfer fails, an error message is printed. The function takes care of handling errors during the 
 * copy process and ensures that the appropriate CUDA memory copy kind is used.
 *
 * @param[out] dst Pointer to the destination memory (either host or device).
 * @param[in] src Pointer to the source memory (either host or device).
 * @param[in] count The number of bytes to copy.
 * @param[in] kind The direction of the copy operation, which can be one of `cudaMemcpyHostToDevice`, 
 *                 `cudaMemcpyDeviceToHost`, or `cudaMemcpyDeviceToDevice`.
 *
 * @note This function does not return a value. If the copy fails, an error message is printed to the console.
 */
void safeCudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    int err = cudaMemcpy(dst, src, count, kind);

    if (err != cudaSuccess) {
        printf("CUDA memcpy failed: %s\n", cudaGetErrorString((cudaError_t)err));
    }
}