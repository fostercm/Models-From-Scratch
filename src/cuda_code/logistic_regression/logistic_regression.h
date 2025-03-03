#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
void fit(const float *X, const float *Y, float *Beta, const int n_samples, const int n_input_features, const int n_classes, const int max_iters, float lr, const float tol);
void predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes);
void _predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes, cublasHandle_t handle);
float cost(const float *Y_pred, const float *Y, const int n_samples, const int n_classes);
}
#endif /* LOGISTIC_REGRESSION_H */
