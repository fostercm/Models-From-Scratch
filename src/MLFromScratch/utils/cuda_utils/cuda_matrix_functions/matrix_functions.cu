/**
 * @file matrix_functions.cu
 * @brief CUDA functions for matrix operations, including printing and populating diagonal elements.
 *
 * This file provides CUDA-accelerated functions for matrix operations, specifically for printing a matrix from device memory to host memory
 * and for populating the diagonal of a matrix using values from a given diagonal vector. 
 * The functions utilize CUDA memory management and kernel launching for efficient computation on GPUs.
 */

#include <cuda_runtime.h>
#include <cuda_matrix_functions/matrix_functions.h>
#include <cuda_memory_functions/memory_functions.h>
#include <cublas_v2.h>
#include <iostream>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * @brief Copies a matrix from device memory to host memory and prints it.
 *
 * This function copies a matrix from the GPU (device memory) to the CPU (host memory) and prints it element by element.
 * It is primarily used for debugging and visualizing matrices stored on the GPU.
 *
 * @param[in] matrix Pointer to the matrix on the GPU.
 * @param[in] rows The number of rows in the matrix.
 * @param[in] cols The number of columns in the matrix.
 *
 * @note This function allocates temporary memory on the host to copy the matrix and print it. The allocated memory is freed after printing.
 */
void printMatrix(const float *matrix, const int rows, const int cols) {

    // Copy matrix from device to host
    float *h_matrix = (float *) malloc(rows * cols * sizeof(float));
    cudaMemcpy(h_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", h_matrix[j * rows + i]);
        }
        printf("\n");
    }
    free(h_matrix);
}

/**
 * @brief CUDA kernel for populating the diagonal of a matrix with the inverse of values from a diagonal vector.
 *
 * This kernel populates the diagonal of a given matrix with the inverse of corresponding values from the `diagonal` vector. 
 * Only non-zero values from the `diagonal` vector are used, with a tolerance of 1e-15 to avoid division by zero. The matrix is assumed to 
 * be square or rectangular, and the diagonal is populated within the valid range (the minimum of the matrix's row and column count).
 *
 * @param[out] matrix The matrix to modify, stored in device memory.
 * @param[in] diagonal The vector of diagonal values stored in device memory.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 *
 * @note This kernel is launched with a grid size determined by the number of diagonal elements and a block size of 256 threads.
 */
__global__ void populateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n) {
    // Get thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Populate diagonal (only within the valid range)
    if (tid < MIN(m, n)) {
        if (diagonal[tid] > 1e-15) {
            matrix[tid * n + tid] = 1.0 / diagonal[tid];
        }
    }
}

/**
 * @brief Launches the CUDA kernel to populate the diagonal of a matrix.
 *
 * This function sets up the appropriate grid and block sizes and launches the `populateDiagonalKernel` to populate the diagonal of 
 * the matrix using values from the provided `diagonal` vector. The kernel is responsible for performing the inversion of the diagonal 
 * values and populating the matrix in device memory.
 *
 * @param[out] matrix The matrix to modify, stored in device memory.
 * @param[in] diagonal The vector of diagonal values stored in device memory.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 *
 * @note This function automatically adjusts the grid size based on the number of diagonal elements to ensure all threads are launched.
 */
void launchPopulateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n) {
    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    populateDiagonalKernel<<<gridSize, blockSize>>>(matrix, diagonal, m, n);
}

__global__ void identityMatrixKernel(float *matrix, const int m, const int n) {
    // Get thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Populate identity matrix
    if (idx < m * n) {

        // Calculate row and column
        int row = idx / n;
        int col = idx % n;

        // Set identity values
        if (row == col) {
            matrix[idx] = 1.0f;
        } else {
            matrix[idx] = 0.0f;
        }
    }
}

void launchIdentityMatrixKernel(float *matrix, const int m, const int n) {
    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    identityMatrixKernel<<<gridSize, blockSize>>>(matrix, m, n);
}

__global__ void scaleValue(float *value, const float scalar) {
    *value *= scalar;
}

void transposeMatrix(float *d_X, const int m, const int n, cublasHandle_t handle) {
    // Initialize error variable
    int err = 0;

    // Allocate memory for transposed matrix
    float *d_X_transposed;
    d_X_transposed = (float*) safeCudaMalloc(m * n * sizeof(float), &err);

    // Define constants
    float alpha = 1.0f;
    float beta = 0.0f;

    // Transpose matrix
    cublasSgeam(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, 
        m, n, 
        &alpha, d_X, n, 
        &beta, d_X_transposed, m,
        d_X_transposed, m
    );

    // Save transposed matrix
    safeCudaMemcpy(d_X, d_X_transposed, m * n * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free memory
    safeCudaFree(d_X_transposed);
}

void standardize(float *d_X, const int m, const int n, cublasHandle_t handle) {    
    // Allocate memory for mean and standard deviation
    int err = 0;
    float *d_mean, *d_std;
    d_mean = (float*) safeCudaMalloc(n * sizeof(float), &err);
    d_std = (float*) safeCudaMalloc(n * sizeof(float), &err);

    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Calculate mean and standard deviation
    meanKernel<<<gridSize, blockSize>>>(d_X, d_mean, m, n);
    stdKernel<<<gridSize, blockSize>>>(d_X, d_mean, d_std, m, n);

    // Standardize matrix
    standardizeKernel<<<gridSize, blockSize>>>(d_X, d_mean, d_std, m, n);

    // Free memory
    safeCudaFree(d_mean);
    safeCudaFree(d_std);
}

__global__ void meanKernel(const float *d_X, float *d_mean, const int m, const int n) {
    // Get thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate mean
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += d_X[i + idx*m];
        }
        d_mean[idx] = sum / m;
    }
}

__global__ void stdKernel(const float *d_X, const float *d_mean, float *d_std, const int m, const int n) {
    // Get thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate standard deviation
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += powf(d_X[idx*m + i] - d_mean[idx], 2);
        }
        d_std[idx] = sqrtf(sum / (m-1));
    }
}

__global__ void standardizeKernel(float *d_X, const float *d_mean, const float *d_std, const int m, const int n) {
    // Get thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Standardize matrix
    if (idx < m * n) {
        int row = idx / m;
        d_X[idx] = (d_X[idx] - d_mean[row]) / d_std[row];
    }
}