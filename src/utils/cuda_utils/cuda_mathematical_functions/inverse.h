#ifndef INVERSE_H
#define INVERSE_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void computeInverse(float *d_A, float *d_A_inv, int n, cublasHandle_t cublasHandle);
void computePseudoinverse(float *A, float *A_inv, const int m, const int n, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH);

#endif /* INVERSE_H */
