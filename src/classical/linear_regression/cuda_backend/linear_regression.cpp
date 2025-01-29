#include "linear_regression_cuda.h"
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
    // Initialize cost and scale factor
    float cost, SCALE_FACTOR;
    cost = 0.0f;
    SCALE_FACTOR = 1.0 / (2 * num_samples);

    // Call device-side function
    costCUDA(Y_pred, Y, &cost, num_samples, num_output_features);

    // Multiply by scale factor and return
    return SCALE_FACTOR * pow(cost, 2);
}