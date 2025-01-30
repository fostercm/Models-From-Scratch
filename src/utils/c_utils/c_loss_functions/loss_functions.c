/**
 * @file loss_functions.c
 * @brief Implements loss functions for machine learning models.
 *
 * This file provides the implementation of loss functions such as 
 * mean squared error (MSE) to evaluate model predictions.
 *
 * @author  Cole Foster
 * @date    2025-01-29
 */

#include <math.h>
#include "loss_functions.h"

/**
 * @brief Computes the Mean Squared Error (MSE) loss.
 *
 * The Mean Squared Error (MSE) is calculated as:
 * MSE = (1\2n) * ||Y_pred - Y||^2
 * where n is the number of samples, Y_pred is the predicted output, 
 * and Y is the ground truth.
 *
 * @param[in] Y_pred Pointer to an array of predicted values.
 * @param[in] Y Pointer to an array of ground truth values.
 * @param[in] num_samples Number of samples in the dataset.
 * @param[in] num_output_features Number of output features per sample.
 * @return Computed mean squared error.
 */
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