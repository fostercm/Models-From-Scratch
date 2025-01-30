#include "pseudoinverse.h"
#include "../cuda_memory_functions/memory_functions.h"
#include "../cuda_matrix_functions/matrix_functions.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Function to compute the pseudoinverse of a matrix
void computePseudoinverse(float *d_A, float *d_A_inv, const int m, const int n, cublasHandle_t cublasH) {
    // Initialize Cusolver
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // Initialize alpha and beta
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Initialize error pointer for safe memory allocation
    int err = 0;

    // Allocate memory for the SVD results
    float *d_U_T, *d_S, *d_V;
    d_U_T = (float*) safeCudaMalloc(m * m * sizeof(float), &err);
    d_S = (float*) safeCudaMalloc(MIN(m, n) * sizeof(float), &err);
    d_V = (float*) safeCudaMalloc(n * n * sizeof(float), &err);

    // Allocate memory for SVD info
    int *devInfo;
    int lwork = 0;
    float *d_work;
    devInfo = (int*) safeCudaMalloc(sizeof(int), &err);
    cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);
    d_work = (float*) safeCudaMalloc(lwork * sizeof(float), &err);

    // Transpose A to make it column-major
    float *d_A_T;
    d_A_T = (float*) safeCudaMalloc(m * n * sizeof(float), &err);
    cublasSgeam(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
        m, n, 
        &alpha, 
        d_A, n, 
        &beta, 
        d_A, m, 
        d_A_T, m
        );

    // Compute SVD
    cusolverDnSgesvd(cusolverH, 'A', 'A', m, n, d_A_T, m, d_S, d_U_T, m, d_V, n, d_work, lwork, NULL, devInfo);

    // Free SVD info
    safeCudaFree(d_work);
    safeCudaFree(devInfo);

    // Initialize S_inv
    float *d_S_inv;
    d_S_inv = (float*) safeCudaMalloc(m * n * sizeof(float), &err);
    cudaMemset(d_S_inv, 0, m * n * sizeof(float));

    // Populate the diagonal of S_inv
    launchPopulateDiagonalKernel(d_S_inv, d_S, m, MIN(m, n));

    // Free S
    safeCudaFree(d_S);

    // Compute VS_inv = V * S_inv and free V & S_inv
    float *d_VS_inv;
    d_VS_inv = (float*) safeCudaMalloc(n * m * sizeof(float), &err);
    cublasSgemm(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
        n, m, n, 
        &alpha, 
        d_V, n, 
        d_S_inv, n, 
        &beta, 
        d_VS_inv, n
        );
    safeCudaFree(d_V);
    safeCudaFree(d_S_inv);

    // Compute A_inv = VS_inv * U_T and free U_T & VS_inv
    cublasSgemm(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_T, 
        n, m, m, 
        &alpha, 
        d_VS_inv, n, 
        d_U_T, m, 
        &beta, 
        d_A_inv, n
        );
    safeCudaFree(d_U_T);
    safeCudaFree(d_VS_inv);
}