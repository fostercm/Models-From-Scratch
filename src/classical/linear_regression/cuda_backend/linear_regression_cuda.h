#ifndef LINEAR_REGRESSION_CUDA_H
#define LINEAR_REGRESSION_CUDA_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void fit();
void predict();
float cost(float *Y_pred, float *Y, int num_samples, int num_output_features);

#ifdef __CUDACC__

    __global__ void kernel_cost();

#endif

#ifdef __cplusplus
}
#endif

#endif /* LINEAR_REGRESSION_CUDA_H */