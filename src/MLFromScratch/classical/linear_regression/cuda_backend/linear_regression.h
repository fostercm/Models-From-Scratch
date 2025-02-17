#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <cuda_runtime.h>

extern "C" {
void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features);
void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features);
float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features);
}

#endif /* LINEAR_REGRESSION_CUDA_H */