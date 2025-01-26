#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linear_regression.h"
#include "pseudoinverse.h"
#include <gsl/gsl_matrix.h>


void fit(double *X, double *Y, double *Beta, int num_samples, int num_input_features, int num_output_features) {

    // Allocate memory for intermediates
    double *inner_product = (double *) malloc(num_input_features * num_input_features * sizeof(double));
    double *inner_product_inv = (double *) malloc(num_input_features * num_samples * sizeof(double));
    double *inner_product_inv_mult = (double *) malloc(num_input_features * num_samples * sizeof(double));
    
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        num_input_features, num_input_features, num_samples,
        1,
        X, num_input_features,
        X, num_input_features,
        0,
        inner_product,num_input_features
        );
    
    // Take the pseudoinverse of the inner product
    compute_pseudoinverse(inner_product, inner_product_inv, num_input_features, num_input_features);

    // Free intermediate
    free(inner_product);

    // Multiply the pseudoinverse and the input matrix
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        num_input_features, num_samples, num_input_features,
        1,
        inner_product_inv, num_input_features,
        X, num_input_features,
        0,
        inner_product_inv_mult, num_samples
        );
    
    // Free intermediate
    free(inner_product_inv);

    // Multiply the running product with Y
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        num_input_features, num_output_features, num_samples,
        1,
        inner_product_inv_mult, num_samples,
        Y, num_output_features,
        0,
        Beta, num_output_features
        );
    
    // Free intermediate
    free(inner_product_inv_mult);
}

void predict(double *X, double *Beta, double *Prediction, int num_samples, int num_input_features, int num_output_features) {
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        num_samples, num_output_features, num_input_features,
        1,
        X, num_input_features,
        Beta, num_output_features,
        0,
        Prediction, num_output_features
        );
}

double cost(double *Y_pred, double *Y, int num_samples, int num_output_features) {

    // Calculate scale factor
    double SCALE_FACTOR = 1.0 / (2 * num_samples);

    // Calculate squared matrix norm
    double cost = 0;
    for (int i=0 ; i<num_samples ; i++) {
        for (int j=0 ; j<num_output_features ; j++) {
            cost += pow(Y_pred[i*num_output_features + j] - Y[i*num_output_features + j], 2);
        }
    }

    // Return scaled cost
    return SCALE_FACTOR * cost;
}