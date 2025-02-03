#include "inverse.h"
#include "pseudoinverse.h"
#include "../cuda_memory_functions/memory_functions.h"
#include "../cuda_matrix_functions/matrix_functions.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

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