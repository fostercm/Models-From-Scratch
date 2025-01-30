#include <math.h>
#include "loss_functions.h"

// Function to calculate the mean squared error
float meanSquaredError(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Calculate scale factor
    const float SCALE_FACTOR = 1.0 / (2 * num_samples);

    // Calculate squared matrix norm
    float cost = 0;
    for (int i=0 ; i<num_samples * num_output_features ; i++) {
        cost += pow(Y_pred[i] - Y[i], 2);
    }

    // Return scaled cost
    return SCALE_FACTOR * cost;
}