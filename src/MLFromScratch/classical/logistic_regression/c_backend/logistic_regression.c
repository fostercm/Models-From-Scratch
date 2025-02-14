/**
 * @file logistic_regression.c
 * @brief Logistic regression model functions for fitting, prediction, and cost calculation.
 *
 * This file contains the implementation of a basic logistic regression model. It includes functions for fitting
 * the model using gradient descent, predicting the output based on input features, and calculating 
 * the cost (cross-entropy loss).
 */

#include "c_loss_functions/loss_functions.h"
#include "c_memory_functions/memory_functions.h"
#include "c_mathematical_functions/activation.h"
#include "logistic_regression.h"
#include <cblas.h>

/**
 * @brief Fit the logistic regression model to the given data.
 *
 * This function trains the logistic regression model using gradient descent.
 * The model weights (Beta) are updated iteratively based on the gradient of the loss.
 *
 * @param[in] X The input matrix of size (n_samples x n_input_features).
 * @param[in] Y The target/output matrix of size (n_samples x n_classes).
 * @param[out] Beta The weights of the model, of size (n_input_features x n_classes).
 * @param[in] n_samples The number of training samples.
 * @param[in] n_input_features The number of input features.
 * @param[in] n_classes The number of output classes.
 * @param[in] max_iters The maximum number of gradient descent iterations.
 * @param[in] lr The learning rate for gradient descent.
 * @param[in] tol The tolerance for early stopping based on the gradient norm.
 */
void fit(const float *X, const float *Y, float *Beta, const int n_samples, const int n_input_features, const int n_classes, const int max_iters, const float lr) {
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
        
        // Update the weights
        cblas_saxpy(n_input_features*n_classes, -lr/n_samples, Gradient, 1, Beta, 1);
    }

    // Free the memory
    safeFree(Gradient);
    safeFree(Prediction);
}

/**
 * @brief Predict the output based on the input features using the trained model.
 *
 * This function predicts the output probabilities using the logistic regression model.
 * For binary classification, it applies the sigmoid activation function.
 * For multiclass classification, it applies the softmax activation function.
 *
 * @param[in] X The input matrix of size (n_samples x n_input_features).
 * @param[in] Beta The model weights, of size (n_input_features x n_classes).
 * @param[out] Prediction The predicted probabilities, of size (n_samples x n_classes).
 * @param[in] n_samples The number of input samples.
 * @param[in] n_input_features The number of input features.
 * @param[in] n_classes The number of output classes.
 */
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

/**
 * @brief Calculate the cost (cross-entropy loss) of the model.
 *
 * This function computes the cross-entropy loss between the predicted values and the actual values.
 * It is used to evaluate the performance of the model after training.
 *
 * @param[in] Y_pred The predicted probabilities, of size (n_samples x n_classes).
 * @param[in] Y The actual target/output matrix, of size (n_samples x n_classes).
 * @param[in] n_samples The number of samples.
 * @param[in] n_classes The number of output classes.
 *
 * @return The cross-entropy loss between the predicted and actual outputs.
 */
float cost(const float *Y_pred, const float *Y, const int n_samples, const int n_classes) {
    // Calculate the cross-entropy loss
    return crossEntropy(Y_pred, Y, n_samples, n_classes);
}