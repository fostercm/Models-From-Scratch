#include "linear_regression_cuda.h"
#include "mathematical_functions/pseudoinverse_cuda.h"
#include "memory_functions/memory_functions.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

void fitCUDA(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
    // Initialize Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialize alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Initialize error pointer for safe memory allocation
    int err = 0;

    // Transfer X to device
    float *d_X;
    d_X = (float*) safeCudaMalloc(num_samples * num_input_features * sizeof(float), &err);
    safeCudaMemcpy(d_X, X, num_samples * num_input_features * sizeof(float), cudaMemcpyHostToDevice);

    // Compute XTX = X^T * X
    float *d_XTX;
    d_XTX = (float*) safeCudaMalloc(num_input_features * num_input_features * sizeof(float), &err);
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        num_input_features, num_input_features, num_samples,
        &alpha,
        d_X, num_input_features, //XT but row major
        d_X, num_input_features, //X but row major
        &beta,
        d_XTX, num_input_features
        );
    
    // Compute XTX inverse and free XTX
    float *d_XTX_inv;
    d_XTX_inv = (float*) safeCudaMalloc(num_input_features * num_input_features * sizeof(float), &err);
    computePseudoinverse(d_XTX, d_XTX_inv, num_input_features, num_input_features, handle);
    safeCudaFree(d_XTX);

    // Compute (XTX)^-1 * XT and free X and XTX_inv
    float *d_XTX_inv_XT;
    d_XTX_inv_XT = (float*) safeCudaMalloc(num_input_features * num_samples * sizeof(float), &err);
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        num_input_features, num_samples, num_input_features,
        &alpha,
        d_XTX_inv, num_input_features, //col major
        d_X, num_input_features, //XT but row major
        &beta,
        d_XTX_inv_XT, num_input_features
        );
    safeCudaFree(d_X);
    safeCudaFree(d_XTX_inv);

    // Transfer Y to device
    float *d_Y;
    d_Y = (float*) safeCudaMalloc(num_samples * num_output_features * sizeof(float), &err);
    safeCudaMemcpy(d_Y, Y, num_samples * num_output_features * sizeof(float), cudaMemcpyHostToDevice);

    // Compute Beta = (XTX)^-1 * XT * Y and free XTX_inv_XT and Y
    float *d_Beta_transpose;
    d_Beta_transpose = (float*) safeCudaMalloc(num_input_features * num_output_features * sizeof(float), &err);
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        num_input_features, num_output_features, num_samples,
        &alpha,
        d_XTX_inv_XT, num_input_features, //Col major
        d_Y, num_output_features, //Row major
        &beta,
        d_Beta_transpose, num_input_features
        );
    safeCudaFree(d_XTX_inv_XT);
    safeCudaFree(d_Y);

    // Transpose Beta and free Beta_T
    float *d_Beta;
    d_Beta = (float*) safeCudaMalloc(num_input_features * num_output_features * sizeof(float), &err);
    cublasSgeam(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        num_output_features, num_input_features, 
        &alpha, 
        d_Beta_transpose, num_input_features, 
        &beta, 
        d_Beta_transpose, num_output_features, 
        d_Beta, num_output_features
        );
    safeCudaFree(d_Beta_transpose);
    
    // Copy data back to CPU
    cudaDeviceSynchronize();
    safeCudaMemcpy(Beta, d_Beta, num_input_features * num_output_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free Beta
    safeCudaFree(d_Beta);

    // Destroy cublas
    cublasDestroy(handle);
}

void predictCUDA(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features) {
    // Initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialize alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Initialize error pointer for safe memory allocation
    int err = 0;

    // Allocate memory on device
    float *d_X, *d_Beta, *d_Prediction_T;
    d_X = (float*) safeCudaMalloc(num_samples * num_input_features * sizeof(float), &err);
    d_Beta = (float*) safeCudaMalloc(num_input_features * num_output_features * sizeof(float), &err);
    d_Prediction_T = (float*) safeCudaMalloc(num_samples * num_output_features * sizeof(float), &err);

    // Copy data to device
    safeCudaMemcpy(d_X, X, num_samples * num_input_features * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Beta, Beta, num_input_features * num_output_features * sizeof(float), cudaMemcpyHostToDevice);

    // Multiply X and Beta then free them
    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T,
        num_samples, num_output_features, num_input_features,
        &alpha,
        d_X, num_input_features,
        d_Beta, num_output_features,
        &beta,
        d_Prediction_T, num_samples
        );
    safeCudaFree(d_X);
    safeCudaFree(d_Beta);

    // Transpose the result to get the final prediction in row-major order
    float *d_Prediction;
    d_Prediction = (float*) safeCudaMalloc(num_samples * num_output_features * sizeof(float), &err);
    cublasSgeam(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        num_output_features, num_samples, 
        &alpha, 
        d_Prediction_T, num_samples, 
        &beta, 
        d_Prediction_T, num_output_features, 
        d_Prediction, num_output_features
        );
    safeCudaFree(d_Prediction_T);

    // Copy data back to CPU
    cudaDeviceSynchronize();
    safeCudaMemcpy(Prediction, d_Prediction, num_samples * num_output_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free remaining memory
    safeCudaFree(d_Prediction);

    // Destroy cublas
    cublasDestroy(handle);
}

void costCUDA(const float *Y_pred, const float *Y, float *cost, const int num_samples, const int num_output_features) {
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