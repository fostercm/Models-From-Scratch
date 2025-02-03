/**
 * @file inverse.cu
 * @brief CUDA implementation of inverse and pseudoinverse computation using LU decomposition and SVD respectively.
 *
 * This file provides a CUDA implementation of the inverse and pseudoinverse of a matrix.
 * It uses LU decomposition and Singular Value Decomposition (SVD) respectively. 
 * It leverages cuBLAS and cuSolver libraries to perform matrix operations efficiently on the GPU. 
 * If the matrix is singular, the pseudoinverse is computed instead of the inverse.
 * Intermediate results are stored and managed on the GPU, and memory is freed after each step to avoid memory leaks.
 */

#include "inverse.h"
#include "../cuda_memory_functions/memory_functions.h"
#include "../cuda_matrix_functions/matrix_functions.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void computeInverse(float *d_A, float *d_A_inv, int n, cublasHandle_t cublasHandle) {
    // Initialize the cuSolver library
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    // Initialize error variable
    int err = 0;

    // Initialize the workspace
    int *d_info, *d_pivot;
    d_info = (int*) safeCudaMalloc(sizeof(int), &err);
    d_pivot = (int*) safeCudaMalloc(n * sizeof(int), &err);

    // Initialize the workspace size
    int lwork = 0;
    int info = 0;
    float *d_work;
    cusolverDnSgetrf_bufferSize(cusolverHandle, n, n, d_A_inv, n, &lwork);
    d_work = (float*) safeCudaMalloc(lwork * sizeof(float), &err);

    // Perform LU decomposition
    cusolverDnSgetrf(cusolverHandle, n, n, d_A_inv, n, d_work, d_pivot, d_info);
    safeCudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    // Check if matrix is singular
    if (info != 0) {
        // Compute pseudoinverse
        computePseudoinverse(d_A, d_A_inv, n, n, cublasHandle, cusolverHandle);
    }
    else {
        // Compute identity matrix
        float *d_identity;
        d_identity = (float*) safeCudaMalloc(n * n * sizeof(float), &err);
        launchIdentityMatrixKernel(d_identity, n, n);

        // Solve the system
        cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, n, n, d_A_inv, n, d_pivot, d_identity, n, d_info);

        // Free memory
        safeCudaFree(d_identity);
    }

    // Free memory
    safeCudaFree(d_info);
    safeCudaFree(d_pivot);
    safeCudaFree(d_work);

    // Destroy the cuSolver handle
    cusolverDnDestroy(cusolverHandle);
}

/**
 * @brief Computes the pseudoinverse of a matrix using CUDA.
 *
 * This function computes the pseudoinverse of a matrix `A` using Singular Value Decomposition (SVD) 
 * and the CUDA cuBLAS and cuSolver libraries. The matrix is transposed to ensure the operations are performed 
 * in a column-major format. The SVD is computed to find the singular values and vectors, and the pseudoinverse 
 * is obtained by inverting the non-zero singular values. The result is stored in `d_A_inv`.
 *
 * @param[in] d_A Pointer to the input matrix `A` stored on the GPU with size `m * n`.
 * @param[out] d_A_inv Pointer to the resulting pseudoinverse matrix, also stored on the GPU with size `n * m`.
 * @param[in] m The number of rows in matrix `A`.
 * @param[in] n The number of columns in matrix `A`.
 * @param[in] cublasH Handle to the cuBLAS library used for matrix operations.
 *
 * @note This function uses cuBLAS for matrix operations and cuSolver for computing the SVD of the matrix.
 *       Memory is allocated and freed for intermediate results throughout the process.
 */
 void computePseudoinverse(float *d_A, float *d_A_inv, const int m, const int n, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH) {
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
    launchPopulateDiagonalKernel(d_S_inv, d_S, m, n);

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