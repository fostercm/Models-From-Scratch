/**
 * @file linear_regression.c
 * @brief Linear regression model functions for fitting, prediction, and cost calculation.
 *
 * This file contains the implementation of a basic linear regression model. It includes functions for fitting
 * the model using ordinary least squares (OLS), predicting the output based on input features, and calculating 
 * the cost (mean squared error).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear_regression.h"
#include "c_mathematical_functions/pseudoinverse.h"
#include "c_loss_functions/loss_functions.h"
#include "c_memory_functions/memory_functions.h"
#include <gsl/gsl_matrix.h>

/**
 * @brief Fit the linear regression model to the given data.
 *
 * This function computes the parameters (Beta) of the linear regression model by solving the normal equation.
 * It uses the formula Beta = (X^T * X)^(-1) * X^T * Y, where X is the input matrix, Y is the target matrix, 
 * and Beta are the model parameters (weights). The pseudoinverse of (X^T * X) is computed to handle the inversion.
 *
 * @param[in] X The input matrix of size (num_samples x num_input_features).
 * @param[in] Y The target/output matrix of size (num_samples x num_output_features).
 * @param[out] Beta The weights of the model, of size (num_input_features x num_output_features).
 * @param[in] num_samples The number of training samples.
 * @param[in] num_input_features The number of features in the input matrix X.
 * @param[in] num_output_features The number of output features in Y.
 */
void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
    // Calculate the inner product of the input matrix and allocate memory for the result
    float *XTX;
    XTX = (float*) safeMalloc(num_input_features * num_input_features * sizeof(float));
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        num_input_features, num_input_features, num_samples,
        1,
        X, num_input_features,
        X, num_input_features,
        0,
        XTX,num_input_features
        );
    
    // Take the pseudoinverse of the inner product and allocate memory for the result
    float *XTX_inverse;
    XTX_inverse = (float*) safeMalloc(num_input_features * num_input_features * sizeof(float));
    computePseudoinverse(XTX, XTX_inverse, num_input_features, num_input_features);

    // Free intermediate
    safeFree(XTX);

    // Multiply the pseudoinverse and the input matrix and allocate memory for the result
    float *XTX_inverse_XT;
    XTX_inverse_XT = (float*) safeMalloc(num_input_features * num_samples * sizeof(float));
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        num_input_features, num_samples, num_input_features,
        1,
        XTX_inverse, num_input_features,
        X, num_input_features,
        0,
        XTX_inverse_XT, num_samples
        );
    
    // Free intermediate
    safeFree(XTX_inverse);

    // Multiply the running product with Y
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        num_input_features, num_output_features, num_samples,
        1,
        XTX_inverse_XT, num_samples,
        Y, num_output_features,
        0,
        Beta, num_output_features
        );
    
    // Free intermediate
    safeFree(XTX_inverse_XT);
}

/**
 * @brief Predict the output based on the input features using the trained model.
 *
 * This function predicts the output using the linear regression model by multiplying the input matrix X
 * with the learned weights Beta. The result is stored in the Prediction matrix.
 *
 * @param[in] X The input matrix of size (num_samples x num_input_features).
 * @param[in] Beta The model weights, of size (num_input_features x num_output_features).
 * @param[out] Prediction The predicted output, of size (num_samples x num_output_features).
 * @param[in] num_samples The number of input samples.
 * @param[in] num_input_features The number of input features in the model.
 * @param[in] num_output_features The number of output features.
 */
void predict(const float *X, const float *Beta, float *Prediction, const int num_samples, const int num_input_features, const int num_output_features) {
    // Multiply the input matrix and weights
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        num_samples, num_output_features, num_input_features,
        1,
        X, num_input_features,
        Beta, num_output_features,
        0,
        Prediction, num_output_features
        );
}

/**
 * @brief Calculate the cost (mean squared error) of the model.
 *
 * This function computes the mean squared error (MSE) between the predicted values and the actual values.
 * It is used to evaluate the performance of the model after training.
 *
 * @param[in] Y_pred The predicted output matrix, of size (num_samples x num_output_features).
 * @param[in] Y The actual target/output matrix, of size (num_samples x num_output_features).
 * @param[in] num_samples The number of samples.
 * @param[in] num_output_features The number of output features.
 *
 * @return The mean squared error between the predicted and actual outputs.
 */
float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Return the mean squared error
    return meanSquaredError(Y_pred, Y, num_samples, num_output_features);
}