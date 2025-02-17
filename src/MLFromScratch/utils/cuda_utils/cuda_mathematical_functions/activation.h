#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cuda_runtime.h>

__global__ void sigmoidKernel(float *d_x, const int rows);
void sigmoid(float *d_x, const int rows);
__global__ void softmaxKernel(float *d_x, const int rows, const int cols);
void softmax(float *d_x, const int rows, const int cols);

#endif /* ACTIVATION_H */
