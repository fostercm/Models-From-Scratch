#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_memory_functions/memory_functions.h"
#include "cuda_mathematical_functions/activation.h"
#include "cuda_loss_functions/loss_functions.h"
#include "logistic_regression.h"

void fit(const float *X, const float *Y, float *Beta, const int n_samples, const int n_input_features, const int n_classes, const int max_iters, const float lr, const float tol) {
    // Initialize cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Initialize error variable
    int err = 0;

    // Initialize device variables
    float *d_X, *d_Y, *d_Beta, *d_Gradient, *d_Predictions;
    d_X = (float*) safeCudaMalloc(n_samples * n_input_features * sizeof(float), &err);
    d_Y = (float*) safeCudaMalloc(n_samples * n_classes * sizeof(float), &err);
    d_Beta = (float*) safeCudaMalloc(n_input_features * n_classes * sizeof(float), &err);
    d_Gradient = (float*) safeCudaMalloc(n_input_features * n_classes * sizeof(float), &err);
    d_Predictions = (float*) safeCudaMalloc(n_samples * n_classes * sizeof(float), &err);

    // Transfer X, Y, and Beta to device
    safeCudaMemcpy(d_X, X, n_samples * n_input_features * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Y, Y, n_samples * n_classes * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Beta, Beta, n_input_features * n_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i=0; i < max_iters; i++) {
        // Predict
        _predict(d_X, d_Beta, d_Predictions, n_samples, n_input_features, n_classes, handle);

        // Check for convergence
        if (i % 1000 == 0) {

        }
    }
}

void predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes) {
    // Initialize cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Initialize error variable
    int err = 0;

    // Initialize device variables
    float *d_X, *d_Beta, *d_Prediction;
    d_X = (float*) safeCudaMalloc(n_samples * n_input_features * sizeof(float), &err);
    d_Beta = (float*) safeCudaMalloc(n_input_features * n_classes * sizeof(float), &err);
    d_Prediction = (float*) safeCudaMalloc(n_samples * n_classes * sizeof(float), &err);

    // Transfer X and Beta to device
    safeCudaMemcpy(d_X, X, n_samples * n_input_features * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Beta, Beta, n_input_features * n_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Predict
    _predict(d_X, d_Beta, d_Prediction, n_samples, n_input_features, n_classes, handle);

    // Transfer prediction to host
    safeCudaMemcpy(Prediction, d_Prediction, n_samples * n_classes * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    safeCudaFree(d_X);
    safeCudaFree(d_Beta);
    safeCudaFree(d_Prediction);

    // Destroy cublas handle
    cublasDestroy(handle);
}

void _predict(const float *d_X, const float *d_Beta, float *d_Prediction, const int n_samples, const int n_input_features, const int n_classes, cublasHandle_t handle) {
    // Initialize alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute X * Beta
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n_classes, n_samples, n_input_features,
        &alpha,
        d_Beta, n_classes, // row major
        d_X, n_input_features, // row major
        &beta,
        d_Prediction, n_classes // row major
    );

    // Apply activation
    if (n_classes == 1) {
        // Sigmoid
        sigmoid(d_Prediction, n_samples);
    } else {
        // Softmax
        softmax(d_Prediction, n_samples, n_classes);
    }
}

float cost(const float *Y_pred, const float *Y, const int n_samples, const int n_classes) {
    // Initialize error variable
    int err = 0;

    // Initialize cost
    float cost = 0.0f;

    // Initialize device variables
    float *d_Y_pred, *d_Y, *d_cost;
    d_Y_pred = (float*) safeCudaMalloc(n_samples * n_classes * sizeof(float), &err);
    d_Y = (float*) safeCudaMalloc(n_samples * n_classes * sizeof(float), &err);
    d_cost = (float*) safeCudaMalloc(sizeof(float), &err);
    safeCudaMemcpy(d_cost, &cost, sizeof(float), cudaMemcpyHostToDevice);

    // Transfer Y_pred and Y to device
    safeCudaMemcpy(d_Y_pred, Y_pred, n_samples * n_classes * sizeof(float), cudaMemcpyHostToDevice);
    safeCudaMemcpy(d_Y, Y, n_samples * n_classes * sizeof(float), cudaMemcpyHostToDevice);

    // Compute cost
    crossEntropy(d_Y_pred, d_Y, d_cost, n_samples, n_classes);

    // Transfer cost to host
    safeCudaMemcpy(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    safeCudaFree(d_Y_pred);
    safeCudaFree(d_Y);
    safeCudaFree(d_cost);

    return cost;
}