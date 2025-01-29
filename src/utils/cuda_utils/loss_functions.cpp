#include "loss_functions_cuda.h"
#include <math.h>

float meanSquaredError(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Initialize cost and scale factor
    float cost, SCALE_FACTOR;
    cost = 0.0f;
    SCALE_FACTOR = 1.0 / (2 * num_samples);

    // Call device-side function
    meanSquaredErrorCUDA(Y_pred, Y, &cost, num_samples, num_output_features);

    // Multiply by scale factor and return
    return SCALE_FACTOR * pow(cost, 2);
}