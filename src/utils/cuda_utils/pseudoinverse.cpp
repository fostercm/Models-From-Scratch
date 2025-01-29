#include "pseudoinverse_cuda.h"
#include "memory_functions.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Function to compute the pseudoinverse of a matrix
void computePseudoinverse(float *A, float *A_inv, const int m, const int n, cublasHandle_t handle) {
    // Establish constants
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocate memory for the SVD results
    float *U_transpose, *S, *V, *superb;
    cudaMalloc(&U_transpose, m * m * sizeof(float));
    cudaMalloc(&S, MIN(m, n) * sizeof(float));
    cudaMalloc(&V, n * n * sizeof(float));
    cudaMalloc(&superb, MIN(m, n) * sizeof(float));

    // Allocate memory for matrix multiplication intermediates
    float *S_inverse, *VS_inverse;
    cudaMalloc(&S_inverse, m * n * sizeof(float));
    cudaMalloc(&VS_inverse, n * m * sizeof(float));

    // Initialize cuSolver
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    // Allocate memory for SVD info
    int *devInfo;
    int lwork = 0;
    float *d_work;

    cudaMalloc(&devInfo, sizeof(int));
    cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));

    // Transpose A to make it column-major
    float *A_transpose;
    cudaMalloc(&A_transpose, m * n * sizeof(float));
    launchTransposeMatrixKernel(A, A_transpose, m, n);

    // Compute SVD
    cusolverDnSgesvd(cusolverH, 'A', 'A', m, n, A_transpose, m, S, U_transpose, m, V, n, d_work, lwork, NULL, devInfo);

    // Check if the SVD was successful
    int info;
    cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    // Free SVD info
    cudaFree(d_work);
    cudaFree(devInfo);

    // Initialize S_inverse
    cudaMemset(S_inverse, 0, m * n * sizeof(float));

    // Populate the diagonal of S_inverse
    launchPopulateDiagonalKernel(S_inverse, S, m, MIN(m, n));

    // Free S
    cudaFree(S);

    // Multiply V and S+ to get VS_inverse
    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        n, m, n, 
        &alpha, 
        V, n, 
        S_inverse, n, 
        &beta, 
        VS_inverse, n
        );
    
    // Free VT and S_Inverse
    cudaFree(V);
    cudaFree(S_inverse);

    // Multiply VS_inverse and U_transpose to get the pseudoinverse
    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        n, m, m, 
        &alpha, 
        VS_inverse, n, 
        U_transpose, m, 
        &beta, 
        A_inv, n
        );

    // Free U and VSI
    cudaFree(U_transpose);
    cudaFree(VS_inverse);
}