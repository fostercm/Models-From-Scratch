#include <cuda_runtime.h>
#include "cuda_utils.h"
#include <iostream>

#define MIN(a, b) (a < b ? a : b)

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

__global__ void transposeMatrixKernel(const float *matrix, float *transposed, const int m, const int n) {
    // Get thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Transpose matrix (only within the valid range)
    if (tid < m * n) {
        int i = tid / n;
        int j = tid % n;
        transposed[j * m + i] = matrix[i * n + j];
    }
}

void launchPopulateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n) {
    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    populateDiagonalKernel<<<gridSize, blockSize>>>(matrix, diagonal, m, n);
}

void launchTransposeMatrixKernel(const float *matrix, float *transposed, const int m, const int n) {
    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (m * n + blockSize - 1) / blockSize;

    // Launch kernel
    transposeMatrixKernel<<<gridSize, blockSize>>>(matrix, transposed, m, n);
}