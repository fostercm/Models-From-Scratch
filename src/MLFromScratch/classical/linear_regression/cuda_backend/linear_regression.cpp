/**
 * @file linear_regression_cuda.cpp
 * @brief Linear regression model functions using CUDA for fitting, prediction, and cost calculation.
 *
 * This file contains the implementation of linear regression functions utilizing CUDA for accelerated
 * computation. It provides functions for fitting the model (training), predicting output based on input data,
 * and calculating the cost (mean squared error) between predictions and actual outputs.
 * The core computations are offloaded to CUDA device functions for enhanced performance.
 */

#include "linear_regression.h"
#include "cuda_loss_functions/loss_functions.h"
#include <math.h>

/**
 * @brief Fit the linear regression model to the data using CUDA.
 *
 * This function calls the device-side function `fitCUDA` to perform the fitting of the linear regression model.
 * The model parameters (Beta) are computed using GPU acceleration for the matrix operations involved in solving
 * the normal equation (Beta = (X^T * X)^(-1) * X^T * Y).
 *
 * @param[in] X The input matrix of size (num_samples x num_input_features).
 * @param[in] Y The target/output matrix of size (num_samples x num_output_features).
 * @param[out] Beta The model parameters (weights), of size (num_input_features x num_output_features).
 * @param[in] num_samples The number of training samples.
 * @param[in] num_input_features The number of input features in the matrix X.
 * @param[in] num_output_features The number of output features in the matrix Y.
 */
void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
    // Call device-side function
    fitCUDA(X, Y, Beta, num_samples, num_input_features, num_output_features);
}

/**
 * @brief Predict the output using the trained model with CUDA.
 *
 * This function calls the device-side function `predictCUDA` to make predictions. It computes the output by
 * multiplying the input matrix X with the trained model parameters Beta, using GPU for efficient computation.
 *
 * @param[in] X The input matrix of size (num_samples x num_input_features).
 * @param[in] Beta The trained model parameters (weights), of size (num_input_features x num_output_features).
 * @param[out] Prediction The predicted output matrix, of size (num_samples x num_output_features).
 * @param[in] num_samples The number of input samples.
 * @param[in] num_input_features The number of input features in the matrix X.
 * @param[in] num_output_features The number of output features.
 */
void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features) {
    // Call device-side function
    predictCUDA(X, Beta, Prediction, num_samples, num_input_features, num_output_features);
}

/**
 * @brief Calculate the cost (mean squared error) of the model using CUDA.
 *
 * This function calls the device-side loss function `meanSquaredError` to calculate the mean squared error
 * between the predicted output (Y_pred) and the actual output (Y). This is used as a measure of model performance.
 *
 * @param[in] Y_pred The predicted output matrix, of size (num_samples x num_output_features).
 * @param[in] Y The actual target/output matrix, of size (num_samples x num_output_features).
 * @param[in] num_samples The number of samples.
 * @param[in] num_output_features The number of output features.
 *
 * @return The mean squared error between the predicted and actual outputs.
 */
float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Call loss function
    return meanSquaredError(Y_pred, Y, num_samples, num_output_features);
}