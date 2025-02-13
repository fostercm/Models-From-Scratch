#include "c_loss_functions/loss_functions.h"
#include "c_memory_functions/memory_functions.h"
#include "c_mathematical_functions/activation.h"
#include "c_matrix_functions/matrix_functions.h"
#include <stdio.h>
#include <cblas.h>

void fit(float *X, float *Y, int n_samples, int n_features, float *weights, int max_iter, float lr) {
    return;
}

void predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes) {

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n_samples, n_classes, n_input_features,
        1,
        X, n_input_features,
        Beta, n_classes,
        0,
        Prediction, n_classes
        );


    if (n_classes == 2) {
        // Apply sigmoid function for binary classification
        sigmoid(Prediction, n_samples, 1);
    }
    else {
        // Apply softmax function for multiclass classification
        softmax(Prediction, n_samples, n_classes);
    }
}

float cost(float *Y_pred, float *Y, const int n_samples, const int n_classes) {
    return crossEntropy(Y_pred, Y, n_samples, n_classes);
}