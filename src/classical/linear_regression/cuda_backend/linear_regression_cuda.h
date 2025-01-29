#ifndef LINEAR_REGRESSION_CUDA_H
#define LINEAR_REGRESSION_CUDA_H

#include <cuda_runtime.h>

extern "C" {

void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features);
void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features);
float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features);

__global__ void vectorDifferenceKernel(const float *a, const float *b, float *result, const int n);
void launchVectorDifference(const float *a, const float *b, float *result, const int n);
}

#endif /* LINEAR_REGRESSION_CUDA_H */