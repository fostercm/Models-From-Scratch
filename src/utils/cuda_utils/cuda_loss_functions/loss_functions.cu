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