/**
 * @file loss_functions.cu
 * @brief CUDA implementation of loss functions.
 *
 * This file contains the CUDA implementation of the mean squared error (MSE) loss function. 
 * It leverages cuBLAS to efficiently compute the cost of MSE on the GPU. 
 * Memory management is handled via safe functions to prevent errors during allocation and copying.
 */

#include "loss_functions.h"
#include "../cuda_memory_functions/memory_functions.h"
#include "../cuda_matrix_functions/matrix_functions.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void meanSquaredError(const float *d_Y_pred, const float *d_Y, float *cost, const int n_samples, const int n_output_features, cublasHandle_t handle) {
    // Initialize alpha and beta
    float alpha = -1.0f;
    float beta = 1.0f;

    // Initialize error pointer for safe memory allocation
    int err = 0;

    // Allocate memory on GPU
    float *d_difference;
    d_difference = (float*) safeCudaMalloc(n_samples * n_output_features * sizeof(float), &err);

    // Calculate the difference and free memory
    cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        n_samples, n_output_features, 
        &alpha, 
        d_Y_pred, n_samples, 
        &beta, 
        d_Y, n_samples, 
        d_difference, n_samples
        );

    // Calculate the norm and free memory
    cublasSnrm2(handle, n_samples * n_output_features, d_difference, 1, cost);
    safeCudaFree(d_difference);

    // Square the cost and scale
    *cost = (*cost * *cost) / (2 * n_samples);
}

void crossEntropy(const float *d_Y_pred, const float *d_Y, float *d_cost, const int n_samples, const int n_classes) {
    // Define the number of threads and blocks 
    int threads_per_block = 256;
    int num_blocks = (n_samples * n_classes + threads_per_block - 1) / threads_per_block;

    // Call the appropriate kernel based on the number of classes
    if (n_classes == 1) {
        binaryCrossEntropyKernel<<<num_blocks, threads_per_block>>>(d_Y_pred, d_Y, d_cost, n_samples);
    } else {
        crossEntropyKernel<<<num_blocks, threads_per_block>>>(d_Y_pred, d_Y, d_cost, n_samples, n_classes);
    }

    // Scale the cost
    float scale = -1.0f / n_samples;
    scaleValue<<<1,1>>>(d_cost, scale);
}

__global__ void crossEntropyKernel(const float *d_Y_pred, const float *d_Y, float *d_cost, const int n_samples, const int n_classes) {
    // Allocate block of shared memory
    __shared__ float shared_memory[256];

    // Get the global and local index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize local cost
    float local_cost = 0.0f;

    // Calculate the cross entropy loss
    if (idx < n_samples * n_classes) {
        local_cost = d_Y[idx] * logf(d_Y_pred[idx] + 1e-10);
    }

    // Store the local cost in shared memory
    shared_memory[tid] = local_cost;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_memory[tid] += shared_memory[tid + stride];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (tid == 0) {
        atomicAdd(d_cost, shared_memory[0]);
    }
}

__global__ void binaryCrossEntropyKernel(const float *d_Y_pred, const float *d_Y, float *d_cost, const int n_samples) {
    // Allocate block of shared memory
    __shared__ float shared_memory[256];

    // Get the global and local index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize local cost
    float local_cost = 0.0f;

    // Calculate the binary cross entropy loss
    if (idx < n_samples) {
        local_cost = d_Y[idx] * logf(d_Y_pred[idx] + 1e-10) + (1 - d_Y[idx]) * logf(1 - d_Y_pred[idx] + 1e-10);
    }

    // Store the local cost in shared memory
    shared_memory[tid] = local_cost;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_memory[tid] += shared_memory[tid + stride];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (tid == 0) {
        atomicAdd(d_cost, shared_memory[0]);
    }
}