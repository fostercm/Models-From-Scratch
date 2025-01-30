/**
 * @file loss_functions.cpp
 * @brief Contains loss function implementations.
 *
 * This file includes various loss functions used in machine learning models.
 * Currently, it contains the implementation of the mean squared error (MSE) loss function.
 */

#include "loss_functions.h"
#include <math.h>

/**
 * @brief Computes the mean squared error (MSE) loss between predicted and true values.
 *
 * This function computes the MSE between the predicted values (`Y_pred`) and the true values (`Y`).
 * It uses a device-side CUDA function `meanSquaredErrorCUDA` to perform the actual computation.
 * The result is then scaled by a factor of \( \frac{1}{2N} \), where \( N \) is the number of samples.
 *
 * The function assumes that `Y_pred` and `Y` are arrays of size `num_samples * num_output_features`.
 *
 * @param[in] Y_pred Pointer to an array of predicted values of size `num_samples * num_output_features`.
 * @param[in] Y Pointer to an array of true values of size `num_samples * num_output_features`.
 * @param[in] num_samples The number of samples in the dataset.
 * @param[in] num_output_features The number of output features per sample.
 * @return The scaled mean squared error loss value.
 */
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