#include <cuda_runtime.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

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