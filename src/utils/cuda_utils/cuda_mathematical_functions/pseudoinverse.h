#ifndef PSEUDOINVERSE_H
#define PSEUDOINVERSE_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void computePseudoinverse(float *A, float *A_inv, const int m, const int n, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH);

#endif /* PSEUDOINVERSE_CUDA_H */
