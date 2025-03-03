#include "principal_component_analysis.h"
#include "cuda_memory_functions/memory_functions.h"
#include "cuda_matrix_functions/matrix_functions.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

int transform(float *X, float *X_transformed, const int n_samples, const int n_features, const int n_components, const float explained_variance_ratio) {
    // Get the number of components to keep
    int n_comp = (n_components > 0) ? n_components : n_features;

    // Initialize error variable
    int err = 0;

    // Initialize the cuBLAS handle
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // Initialize the cuSOLVER handle
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    // Move the data to the GPU
    float *d_X, *d_X_transformed;
    d_X = (float*) safeCudaMalloc(n_samples * n_features * sizeof(float), &err);
    d_X_transformed = (float*) safeCudaMalloc(n_samples * n_features * sizeof(float), &err);
    safeCudaMemcpy(d_X, X, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);

    // Standardize the matrix
    standardize(d_X, n_samples, n_features, cublasHandle);

    // Copy the standardized matrix to the transformed matrix
    safeCudaMemcpy(d_X_transformed, d_X, n_samples * n_features * sizeof(float), cudaMemcpyDeviceToDevice);

    // Allocate memory for the SVD results
    float *d_U, *d_S, *d_V_T;
    d_U = (float*) safeCudaMalloc(n_samples * n_samples * sizeof(float), &err);
    d_S = (float*) safeCudaMalloc(MIN(n_samples,n_features) * sizeof(float), &err);
    d_V_T = (float*) safeCudaMalloc(n_features * n_features * sizeof(float), &err);

    // Allocate memory for SVD info
    int *devInfo;
    int lwork = 0;
    float *d_work;
    devInfo = (int*) safeCudaMalloc(sizeof(int), &err);
    cusolverDnSgesvd_bufferSize(cusolverHandle, n_samples, n_features, &lwork);
    d_work = (float*) safeCudaMalloc(lwork * sizeof(float), &err);

    // Perform SVD
    cusolverDnSgesvd(cusolverHandle, 'A', 'A', n_samples, n_features, d_X_transformed, n_samples, d_S, d_U, n_samples, d_V_T, n_features, d_work, lwork, NULL, devInfo);

    // If explained variance ratio is set, calculate the number of components to keep
    if (explained_variance_ratio > 0.0f) {
        // Copy S to the host
        float *S;
        S = (float*) malloc(n_features * sizeof(float));
        safeCudaMemcpy(S, d_S, n_features * sizeof(float), cudaMemcpyDeviceToHost);

        // Calculate the total variance
        float total_variance = 0.0f;
        float *d_variance;
        d_variance = (float*) safeCudaMalloc(sizeof(float), &err);
        _computeVarianceKernel<<<1, 256>>>(d_S, d_variance, n_features);
        safeCudaMemcpy(&total_variance, d_variance, sizeof(float), cudaMemcpyDeviceToHost);
        safeCudaFree(d_variance);

        // Calculate the number of components to keep
        float variance_ratio = 0.0f;
        for (int i = 0; i < n_features; i++) {
            variance_ratio += S[i] * S[i];
            if (variance_ratio / total_variance >= explained_variance_ratio) {
                n_comp = i + 1;
                break;
            }
        }
    }

    // Calculate the transformed matrix
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        n_samples, n_comp, n_features,
        &alpha, d_X, n_samples,
        d_V_T, n_features,
        &beta, d_X_transformed, n_samples
    );

    // Free memory
    safeCudaFree(d_X);
    safeCudaFree(d_U);
    safeCudaFree(d_S);
    safeCudaFree(d_V_T);
    safeCudaFree(d_work);
    safeCudaFree(devInfo);

    // Copy the transformed matrix back to the host
    safeCudaMemcpy(X_transformed, d_X_transformed, n_samples * n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    safeCudaFree(d_X_transformed);

    // Destroy handles
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    // Return the number of components
    return n_comp;
}

__global__ void _computeVarianceKernel(const float *d_S, float *d_variance, const int n_features) {
    // Allocate shared memory
    __shared__ float shared_memory[256];

    // Get thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate variance
    if (idx < n_features) {
        shared_memory[threadIdx.x] = d_S[idx] * d_S[idx];
    }

    // Sync threads
    __syncthreads();

    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write to global memory
    if (threadIdx.x == 0) {
        atomicAdd(d_variance, shared_memory[0]);
    }
}