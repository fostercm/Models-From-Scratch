#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear_regression_c.h"
#include "pseudoinverse.h"
#include <gsl/gsl_matrix.h>

// Function to fit the model
void fit(float *X, float *Y, float *Beta, int num_samples, int num_input_features, int num_output_features) {

    // Allocate memory for intermediates
    float *XTX = (float *) malloc(num_input_features * num_input_features * sizeof(float));
    float *XTX_inverse = (float *) malloc(num_input_features * num_samples * sizeof(float));
    float *XTX_inverse_XT = (float *) malloc(num_input_features * num_samples * sizeof(float));
    
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        num_input_features, num_input_features, num_samples,
        1,
        X, num_input_features,
        X, num_input_features,
        0,
        XTX,num_input_features
        );
    
    // Take the pseudoinverse of the inner product
    computePseudoinverse(XTX, XTX_inverse, num_input_features, num_input_features);

    // Free intermediate
    free(XTX);

    // Multiply the pseudoinverse and the input matrix
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
void predict(float *X, float *Beta, float *Prediction, int num_samples, int num_input_features, int num_output_features) {
    
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
float cost(float *Y_pred, float *Y, int num_samples, int num_output_features) {

    // Calculate scale factor
    float SCALE_FACTOR = 1.0 / (2 * num_samples);

    // Calculate squared matrix norm
    float cost = 0;
    for (int i=0 ; i<num_samples ; i++) {
        for (int j=0 ; j<num_output_features ; j++) {
            cost += pow(Y_pred[i*num_output_features + j] - Y[i*num_output_features + j], 2);
        }
    }

    // Return scaled cost
    return SCALE_FACTOR * cost;
}