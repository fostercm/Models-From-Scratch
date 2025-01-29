#include "linear_regression_cuda.h"
#include "pseudoinverse_cuda.h"
#include "cuda_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

void print_matrix(const float *matrix, const int rows, const int cols) {
    float *h_matrix = (float *) malloc(rows * cols * sizeof(float));
    cudaMemcpy(h_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }

    free(h_matrix);
}

void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
    // Initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Establish constants
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Initialize GPU memory
    float *d_X, *d_Y;
    cudaMalloc(&d_X, num_samples * num_input_features * sizeof(float));
    cudaMalloc(&d_Y, num_samples * num_output_features * sizeof(float));

    // Error handling
    if (d_X == NULL || d_Y == NULL) {
        fprintf(stderr, "Error allocating memory on GPU\n");
        exit(EXIT_FAILURE);
    }

    // Copy data to GPU
    cudaMemcpy(d_X, X, num_samples * num_input_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, num_samples * num_output_features * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the inner product of the input matrix and allocate memory for the result
    float *XTX;
    cudaMalloc(&XTX, num_input_features * num_input_features * sizeof(float));
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        num_input_features, num_input_features, num_samples,
        &alpha,
        d_X, num_input_features, //XT but row major
        d_X, num_input_features, //X but row major
        &beta,
        XTX, num_input_features
        );
    
    // Take the pseudoinverse of the inner product and allocate memory for the result
    float *XTX_inverse;
    cudaMalloc(&XTX_inverse, num_input_features * num_input_features * sizeof(float));
    computePseudoinverse(XTX, XTX_inverse, num_input_features, num_input_features);

    // Free memory on GPU
    cudaFree(XTX);

    // Multiply the pseudoinverse with the input matrix and allocate memory for the result
    float *XTX_inverse_XT;
    cudaMalloc(&XTX_inverse_XT, num_input_features * num_samples * sizeof(float));
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        num_input_features, num_samples, num_input_features,
        &alpha,
        XTX_inverse, num_input_features, //col major
        d_X, num_input_features, //XT but row major
        &beta,
        XTX_inverse_XT, num_input_features
        );

    // Free memory on GPU
    cudaFree(d_X);
    cudaFree(XTX_inverse);

    // Multiply the result with the output matrix and allocate memory for the result
    float *d_Beta_transpose;
    cudaMalloc(&d_Beta_transpose, num_input_features * num_output_features * sizeof(float));
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        num_input_features, num_output_features, num_samples,
        &alpha,
        XTX_inverse_XT, num_input_features, //Col major
        d_Y, num_output_features, //Row major
        &beta,
        d_Beta_transpose, num_input_features
        );

    // Free memory on GPU
    cudaFree(XTX_inverse_XT);
    cudaFree(d_Y);
    
    // Transpose the result to get the final weights in row-major order
    float *d_Beta;
    cudaMalloc(&d_Beta, num_input_features * num_output_features * sizeof(float));
    cublasSgeam(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        num_output_features, num_input_features, 
        &alpha, 
        d_Beta_transpose, num_input_features, 
        &beta, 
        d_Beta_transpose, num_output_features, 
        d_Beta, num_output_features
        );
    
    // Copy data back to CPU
    cudaDeviceSynchronize();
    cudaMemcpy(Beta, d_Beta, num_input_features * num_output_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_Beta);

    // Destroy cublas
    cublasDestroy(handle);
}

void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features) {
    
    // Initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on GPU
    float *d_X, *d_Beta, *d_Prediction_transpose;
    cudaMalloc(&d_X, num_samples * num_input_features * sizeof(float));
    cudaMalloc(&d_Beta, num_input_features * num_output_features * sizeof(float));
    cudaMalloc(&d_Prediction_transpose, num_samples * num_output_features * sizeof(float));

    // Error handling
    if (d_X == NULL || d_Beta == NULL || d_Prediction_transpose == NULL) {
        fprintf(stderr, "Error allocating memory on GPU\n");
        exit(EXIT_FAILURE);
    }

    // Copy data to GPU
    cudaMemcpy(d_X, X, num_samples * num_input_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Beta, Beta, num_input_features * num_output_features * sizeof(float), cudaMemcpyHostToDevice);

    // Multiply the input matrix with the weights
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T,
        num_samples, num_output_features, num_input_features,
        &alpha,
        d_X, num_input_features,
        d_Beta, num_output_features,
        &beta,
        d_Prediction_transpose, num_samples
        );

    // Transpose the result to get the final prediction in row-major order
    float *d_Prediction;
    cudaMalloc(&d_Prediction, num_samples * num_output_features * sizeof(float));
    cublasSgeam(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        num_output_features, num_samples, 
        &alpha, 
        d_Prediction_transpose, num_samples, 
        &beta, 
        d_Prediction_transpose, num_output_features, 
        d_Prediction, num_output_features
        );

    // Copy data back to CPU
    cudaDeviceSynchronize();
    cudaMemcpy(Prediction, d_Prediction, num_samples * num_output_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_X);
    cudaFree(d_Beta);
    cudaFree(d_Prediction_transpose);
    cudaFree(d_Prediction);

    // Destroy cublas
    cublasDestroy(handle);
}

float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Get the total number of elements
    const int n = num_samples * num_output_features;

    // Allocate memory on GPU
    float *d_Y_pred, *d_Y, *d_difference, *cost;
    cudaMalloc(&d_Y_pred, n * sizeof(float));
    cudaMalloc(&d_Y, n * sizeof(float));
    cudaMalloc(&d_difference, n * sizeof(float));
    *cost = 0.0f;

    // Copy data to GPU
    cudaMemcpy(d_Y_pred, Y_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the cost
    const float alpha = -1.0f;
    const float beta = 1.0f;

    // Initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgeam(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        num_samples, num_output_features, 
        &alpha, 
        d_Y_pred, num_samples, 
        &beta, 
        d_Y, num_samples, 
        d_difference, num_samples
        );
    
    cublasSnrm2(handle, n, d_difference, 1, cost);

    // Destroy cublas
    cublasDestroy(handle);

    // Free memory on GPU
    cudaFree(d_Y_pred);
    cudaFree(d_Y);
    cudaFree(d_difference);

    // Multiply by scale factor and return
    float SCALE_FACTOR = 1.0 / (2 * num_samples);
    return SCALE_FACTOR * pow(*cost, 2);
}