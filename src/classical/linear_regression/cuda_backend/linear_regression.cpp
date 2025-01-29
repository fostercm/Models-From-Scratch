#include "linear_regression_cuda.h"
#include "loss_functions_cuda.h"
#include <math.h>

void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
    // Call device-side function
    fitCUDA(X, Y, Beta, num_samples, num_input_features, num_output_features);
}

void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features) {
    // Call device-side function
    predictCUDA(X, Beta, Prediction, num_samples, num_input_features, num_output_features);
}

float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Call loss function
    return meanSquaredError(Y_pred, Y, num_samples, num_output_features);
}