#include "loss_functions.h"
#include "../cuda_memory_functions/memory_functions.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

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