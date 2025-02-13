#include "c_loss_functions/loss_functions.h"
#include "c_memory_functions/memory_functions.h"
#include <stdio.h>

void fit(float *X, float *Y, int n_samples, int n_features, float *weights, int max_iter, float lr) {
    return;
}

void predict(float *X, float *weights, int n_samples, int n_features, float *preds) {
    return;
}

float cost(float *Y_pred, float *Y, int n_samples, int n_classes) {
    return crossEntropy(Y_pred, Y, n_samples, n_classes);
}