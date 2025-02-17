/**
 * @file linear_regression.cu
 * @brief Linear regression model functions using CUDA for fitting, prediction, and cost calculation.
 *
 * This file contains the CUDA-accelerated implementation of linear regression model fitting, prediction, and
 * cost computation. The matrix operations are offloaded to the GPU using cuBLAS and custom CUDA functions.
 * The core functions involve fitting the model (training), making predictions, and calculating the cost (mean squared error).
 */

#include "linear_regression.h"
#include "cuda_mathematical_functions/inverse.h"
#include "cuda_memory_functions/memory_functions.h"
#include "cuda_loss_functions/loss_functions.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * @brief Fit the linear regression model using CUDA.
 *
 * This function performs the model fitting (training) on the GPU. It computes the Beta coefficients using the
 * normal equation Beta = (X^T * X)^(-1) * X^T * Y, with CUDA-accelerated matrix multiplication via cuBLAS,
 * and the pseudoinverse computed on the GPU.
 *
 * @param[in] X The input matrix (num_samples x num_input_features).
 * @param[in] Y The target/output matrix (num_samples x num_output_features).
 * @param[out] Beta The computed model parameters (weights), (num_input_features x num_output_features).
 * @param[in] num_samples The number of samples in the dataset.
 * @param[in] num_input_features The number of features in the input data (X).
 * @param[in] num_output_features The number of output features (Y).
 */
void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
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
    computeInverse(d_XTX, d_XTX_inv, num_input_features, handle);
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

/**
 * @brief Predict the output using the trained linear regression model with CUDA.
 *
 * This function computes the predicted output by multiplying the input matrix X with the learned model parameters Beta,
 * using cuBLAS for efficient matrix multiplication on the GPU.
 *
 * @param[in] X The input matrix (num_samples x num_input_features).
 * @param[in] Beta The trained model parameters (weights), (num_input_features x num_output_features).
 * @param[out] Prediction The predicted output matrix, (num_samples x num_output_features).
 * @param[in] num_samples The number of samples in the dataset.
 * @param[in] num_input_features The number of input features in the dataset.
 * @param[in] num_output_features The number of output features.
 */
void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features) {
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

float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Initialize cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialize error pointer for safe memory allocation
    int err = 0;

    // Allocate memory for cost and on device
    float *d_Y_pred, *d_Y;
    float cost;
    d_Y_pred = (float*) safeCudaMalloc(num_samples * num_output_features * sizeof(float), &err);
    d_Y = (float*) safeCudaMalloc(num_samples * num_output_features * sizeof(float), &err);

    // Copy data to device
    safeCudaMemcpy(d_Y_pred, Y_pred, num_samples * num_output_features * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Y, Y, num_samples * num_output_features * sizeof(float), cudaMemcpyHostToDevice);

    // Call loss function
    meanSquaredError(d_Y_pred, d_Y, &cost, num_samples, num_output_features, handle);

    // Free memory
    safeCudaFree(d_Y_pred);
    safeCudaFree(d_Y);

    // Destroy cublas
    cublasDestroy(handle);

    return cost;
}