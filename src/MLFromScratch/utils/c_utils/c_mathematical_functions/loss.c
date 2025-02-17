/**
 * @file loss_functions.c
 * @brief Implements loss functions for machine learning models.
 *
 * This file contains the implementation of various loss functions commonly used in machine learning models 
 * to evaluate their performance. Currently, it includes functions like Mean Squared Error (MSE) and 
 * Cross Entropy, which are used for regression and classification tasks, respectively.
 *
 * The MSE loss function is typically used for regression problems, where the model predicts continuous values, 
 * and the goal is to minimize the squared difference between the predicted values and the true values.
 * The Cross Entropy loss function is typically used for classification problems, where the model outputs probabilities 
 * and the goal is to minimize the difference between the predicted class probabilities and the true class labels.
 *
 * @author  Cole Foster
 * @date    2025-01-29
 */


#include <math.h>
#include <stdio.h>
#include "loss.h"
#include <omp.h>

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
    #pragma omp parallel for reduction(+:cost)
    for (int i=0 ; i<num_samples * num_output_features ; i++) {
        cost += pow(Y_pred[i] - Y[i], 2);
    }

    // Return scaled cost
    return SCALE_FACTOR * cost;
}

/**
 * @brief Computes the Cross-Entropy loss.
 *
 * The Cross-Entropy loss is commonly used for classification problems, particularly when the model 
 * outputs class probabilities. It is calculated as:
 * 
 * CrossEntropy = -(1/n) * sum(Y * log(Y_pred))
 * where n is the number of samples, Y_pred is the predicted probability distribution for each sample, 
 * and Y is the true class labels (one-hot encoded).
 *
 * This function computes the loss for multi-class classification tasks by iterating over all samples 
 * and classes. The loss is calculated as the negative log likelihood of the true class for each sample.
 *
 * @param[in] Y_pred Pointer to an array of predicted probabilities, where each element represents 
 *                   the predicted probability of a particular class for each sample.
 * @param[in] Y Pointer to an array of true class labels (one-hot encoded), where each element 
 *              corresponds to the true label of a particular sample.
 * @param[in] num_samples The number of samples in the dataset.
 * @param[in] num_classes The number of classes in the classification task.
 * @return Computed cross-entropy loss.
 */
float crossEntropy(const float *Y_pred, const float *Y, const int num_samples, const int num_classes) {
    // Calculate scale factor
    const float SCALE_FACTOR = -1.0 / num_samples;

    // Calculate cross entropy
    float cost = 0;

    if (num_classes == 2) {
        // Binary classification
        #pragma omp parallel for reduction(+:cost)
        for (int i=0 ; i<num_samples ; i++) {
            cost += Y[i] * log(Y_pred[i] + 1e-9) + (1 - Y[i]) * log(1 - Y_pred[i] + 1e-9);
        }
    }
    else {
        // Multi-class classification
        // #pragma omp parallel for reduction(+:cost)
        for (int i=0 ; i<num_samples * num_classes ; i++) {
            cost += Y[i] * log(Y_pred[i] + 1e-9);
        }
    }

    // Return scaled cost
    return SCALE_FACTOR * cost;
}