#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
    void crossEntropy(const float *d_Y_pred, const float *d_Y, float *d_cost, const int n_samples, const int n_classes);
    void meanSquaredError(const float *d_Y_pred, const float *d_Y, float *d_cost, const int n_samples, const int n_output_features, cublasHandle_t handle);
} 
__global__ void crossEntropyKernel(const float *Y_pred, const float *Y, float *cost, const int num_samples, const int num_output_features);
__global__ void binaryCrossEntropyKernel(const float *d_Y_pred, const float *d_Y, float *d_cost, const int n_samples);
#endif /* LOSS_FUNCTIONS_H */
