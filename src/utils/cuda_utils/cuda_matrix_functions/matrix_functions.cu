#include <cuda_runtime.h>
#include <matrix_functions.h>
#include <iostream>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void printMatrixCUDA(const float *matrix, const int rows, const int cols) {

    // Copy matrix from device to host
    float *h_matrix = (float *) malloc(rows * cols * sizeof(float));
    cudaMemcpy(h_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    free(h_matrix);
}

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

void launchPopulateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n) {
    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    populateDiagonalKernel<<<gridSize, blockSize>>>(matrix, diagonal, m, n);
}