#ifndef INVERSE_H
#define INVERSE_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void computeInverse(float *d_A, float *d_A_inv, int n, cublasHandle_t cublasHandle);

#endif /* INVERSE_H */
