#include "c_loss_functions/loss_functions.h"
#include "c_memory_functions/memory_functions.h"
#include "c_mathematical_functions/activation.h"
#include "c_matrix_functions/matrix_functions.h"
#include "logistic_regression.h"
#include <stdio.h>
#include <cblas.h>

void fit(const float *X, const float *Y, float *Beta, const int n_samples, const int n_input_features, const int n_classes, const int max_iters, const float lr, const float tol) {
    // Initialize the gradient and prediction matrices
    float *Gradient = safeMalloc(n_input_features * n_classes * sizeof(float));
    float *Prediction = safeMalloc(n_samples * n_classes * sizeof(float));
    
    // Loop through the number of iterations
    for (int i=0; i<max_iters; i++) {
        // Make a prediction
        predict(X, Beta, Prediction, n_samples, n_input_features, n_classes);

        // Calculate the difference between the prediction and the true values (in place)
        cblas_saxpy(n_samples*n_classes, -1, Y, 1, Prediction, 1);

        // Multiply the transpose of the input matrix by the difference (compute the gradient)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            n_input_features, n_classes, n_samples,
            1,
            X, n_input_features,
            Prediction, n_classes,
            0,
            Gradient, n_classes
            );

        // Divide the gradient by the number of samples
        cblas_sscal(n_input_features*n_classes, 1.0/n_samples, Gradient, 1);

        // Check if the norm of the gradient is less than the tolerance
        if (cblas_snrm2(n_input_features*n_classes, Gradient, 1) < tol) {
            break;
        }
        
        // Update the weights
        cblas_saxpy(n_input_features*n_classes, -lr, Gradient, 1, Beta, 1);
    }

    // Free the memory
    safeFree(Gradient);
    safeFree(Prediction);
}

void predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes) {
    // Multiply the input matrix by the weights
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n_samples, n_classes, n_input_features,
        1,
        X, n_input_features,
        Beta, n_classes,
        0,
        Prediction, n_classes
        );

    if (n_classes == 1) {
        // Apply sigmoid function for binary classification
        sigmoid(Prediction, n_samples, 1);
    }
    else {
        // Apply softmax function for multiclass classification
        softmax(Prediction, n_samples, n_classes);
    }
}

float cost(const float *Y_pred, const float *Y, const int n_samples, const int n_classes) {
    // Calculate the cross-entropy loss
    return crossEntropy(Y_pred, Y, n_samples, n_classes);
}