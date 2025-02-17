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

/**
 * @brief Computes the mean squared error (MSE) loss using CUDA.
 *
 * This function computes the MSE loss between the predicted values (`Y_pred`) and the true values (`Y`)
 * on the GPU using cuBLAS operations. It calculates the squared difference between the predictions 
 * and the true values, then computes the Euclidean norm of the result.
 *
 * The function assumes that `Y_pred` and `Y` are arrays of size `num_samples * num_output_features`.
 *
 * Memory for GPU operations is dynamically allocated, and memory is freed after each step to avoid 
 * memory leaks. A handle for cuBLAS is initialized and destroyed within this function.
 *
 * @param[in] Y_pred Pointer to the predicted values array of size `num_samples * num_output_features`.
 * @param[in] Y Pointer to the true values array of size `num_samples * num_output_features`.
 * @param[out] cost Pointer to the scalar result, which will store the computed MSE cost.
 * @param[in] num_samples The number of samples in the dataset.
 * @param[in] num_output_features The number of output features per sample.
 */
void meanSquaredErrorCUDA(const float *Y_pred, const float *Y, float *cost, const int num_samples, const int num_output_features) {
    // Initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialize alpha and beta
    float alpha = -1.0f;
    float beta = 1.0f;

    // Get the total number of elements
    const int n = num_samples * num_output_features;

    // Initialize error pointer for safe memory allocation
    int err = 0;

    // Allocate memory on GPU
    float *d_Y_pred, *d_Y, *d_difference;
    d_Y_pred = (float*) safeCudaMalloc(n * sizeof(float), &err);
    d_Y = (float*) safeCudaMalloc(n * sizeof(float), &err);
    d_difference = (float*) safeCudaMalloc(n * sizeof(float), &err);

    // Copy data to GPU
    safeCudaMemcpy(d_Y_pred, Y_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Y, Y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the difference and free memory
    cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        num_samples, num_output_features, 
        &alpha, 
        d_Y_pred, num_samples, 
        &beta, 
        d_Y, num_samples, 
        d_difference, num_samples
        );
    safeCudaFree(d_Y_pred);
    safeCudaFree(d_Y);
    
    // Calculate the norm, send to host, and free memory
    cublasSnrm2(handle, n, d_difference, 1, cost);
    safeCudaFree(d_difference);

    // Destroy cublas
    cublasDestroy(handle);
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