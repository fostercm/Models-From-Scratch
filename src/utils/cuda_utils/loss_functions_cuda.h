#ifndef LOSS_FUNCTIONS_CUDA_H
#define LOSS_FUNCTIONS_CUDA_H

#include <cuda_runtime.h>

extern "C" {
    float meanSquaredError(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features);
    void meanSquaredErrorCUDA(const float *Y_pred, const float *Y, float *cost, const int num_samples, const int num_output_features);
}

#endif /* LOSS_FUNCTIONS_CUDA_H */
