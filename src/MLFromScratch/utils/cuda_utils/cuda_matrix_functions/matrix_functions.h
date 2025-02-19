#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

void printMatrix(const float *matrix, const int rows, const int cols);
__global__ void populateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n);
void launchPopulateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n);
__global__ void identityMatrixKernel(float *matrix, const int m, const int n);
void launchIdentityMatrixKernel(float *matrix, const int m, const int n);
__global__ void scaleValue(float *value, const float scalar);
void standardize(float *X, const int m, const int n, cublasHandle_t handle);
void transposeMatrix(float *d_X, const int m, const int n, cublasHandle_t handle);
__global__ void meanKernel(const float *d_X, float *d_mean, const int m, const int n);
__global__ void stdKernel(const float *d_X, const float *d_mean, float *d_std, const int m, const int n);
__global__ void standardizeKernel(float *d_X, const float *d_mean, const float *d_std, const int m, const int n);

#endif /* MATRIX_FUNCTIONS_H */
