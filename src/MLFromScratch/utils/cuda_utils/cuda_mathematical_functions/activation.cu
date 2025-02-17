#include <cuda_runtime.h>
#include "activation.h"

__global__ void sigmoidKernel(float *d_x, const int rows) {
    // Get the thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        // Use different forms of the sigmoid function for numerical stability
        if (d_x[idx] >= 0)
            d_x[idx] = 1 / (1 + expf(-d_x[idx]));
        else
            d_x[idx] = expf(d_x[idx]) / (1 + expf(d_x[idx]));
    }
}

void sigmoid(float *d_x, const int rows) {
    // Calculate the number of blocks needed
    int numBlocks = (rows + 256 - 1) / 256;

    // Call the kernel
    sigmoidKernel<<<numBlocks, 256>>>(d_x, rows);
}

__global__ void softmaxKernel(float *d_x, const int rows, const int cols) {
    // Get the thread id
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        // Find the maximum value in the row
        float max_val = d_x[idx * cols];
        for (int i = 1; i < cols; i++) {
            if (d_x[idx * cols + i] > max_val)
                max_val = d_x[idx * cols + i];
        }

        // Subtract the maximum value from each element in the row
        for (int i = 0; i < cols; i++) {
            d_x[idx * cols + i] -= max_val;
        }

        // Calculate the sum of the row
        float sum = 0;
        for (int i = 0; i < cols; i++) {
            d_x[idx * cols + i] = expf(d_x[idx * cols + i]);
            sum += d_x[idx * cols + i];
        }

        // Normalize the row
        for (int i = 0; i < cols; i++) {
            d_x[idx * cols + i] /= sum;
        }
    }
}

void softmax(float *d_x, const int rows, const int cols) {
    // Calculate the number of blocks needed
    int numBlocks = (rows + 256 - 1) / 256;

    // Call the kernel
    softmaxKernel<<<numBlocks, 256>>>(d_x, rows, cols);
}