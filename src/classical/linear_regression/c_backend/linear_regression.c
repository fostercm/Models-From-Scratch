#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear_regression.h"
#include "c_mathematical_functions/pseudoinverse.h"
#include "c_loss_functions/loss_functions.h"
#include <gsl/gsl_matrix.h>

// Function to fit the model
void fit(const float *X, const float *Y, float *Beta, const int num_samples, const int num_input_features, const int num_output_features) {
    // Calculate the inner product of the input matrix and allocate memory for the result
    float *XTX = (float *) malloc(num_input_features * num_input_features * sizeof(float));
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
    float *XTX_inverse = (float *) malloc(num_input_features * num_samples * sizeof(float));
    computePseudoinverse(XTX, XTX_inverse, num_input_features, num_input_features);

    // Free intermediate
    free(XTX);

    // Multiply the pseudoinverse and the input matrix and allocate memory for the result
    float *XTX_inverse_XT = (float *) malloc(num_input_features * num_samples * sizeof(float));
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
    free(XTX_inverse);

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
    free(XTX_inverse_XT);
}

// Function to predict the output using weights
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

// Function to calculate the cost
float cost(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features) {
    // Return the mean squared error
    return meanSquaredError(Y_pred, Y, num_samples, num_output_features);
}