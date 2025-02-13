/**
 * @file activation.c
 * @brief Implementation of common activation functions: Sigmoid and Softmax.
 *
 * This file provides parallelized implementations of the sigmoid and softmax 
 * activation functions using OpenMP for improved performance on large arrays.
 * 
 * Functions Included:
 * - sigmoid(): Applies the sigmoid function to an array.
 * - softmax(): Applies the softmax function to an array.
 *
 * Numerical Stability:
 * Both functions are implemented with techniques to improve numerical stability:
 * - Sigmoid uses different formulas depending on the sign of the input.
 * - Softmax subtracts the maximum value from inputs before exponentiation.
 *
 * Parallelization:
 * OpenMP is used to parallelize operations for both functions:
 * - `#pragma omp parallel for`: Loops are distributed across threads.
 * - `reduction`: Thread-safe aggregation for max and sum operations.
 *
 * Usage:
 * ```
 * #include "activation.h"
 * double arr[] = {1.0, 2.0, 3.0};
 * int size = 3;
 * sigmoid(arr, size); // Applies sigmoid in-place
 * softmax(arr, size); // Applies softmax in-place
 * ```
 *
 * Dependencies:
 * - math.h   : For mathematical operations (exp, INFINITY).
 * - omp.h    : For OpenMP parallelization.
 *
 * @author Cole Foster
 * @version 1.0
 * @date February 13, 2025
 */

#include <math.h>
#include <omp.h>

/**
 * @brief Applies the sigmoid function to each element of an array in parallel.
 *
 * The sigmoid function is defined as:
 *   sigmoid(x) = 1 / (1 + exp(-x))
 *
 * To improve numerical stability, it uses two different forms of the sigmoid function:
 *   - When x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
 *   - When x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
 *
 * @param x Pointer to an array of doubles representing input values.
 * @param size The number of elements in the array.
 *
 * @note The function modifies the input array in place.
 * @note OpenMP is used to parallelize the computation for faster performance.
 */
void sigmoid(float *x, const int rows, const int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows*cols; i++) {
        // Use different forms of the sigmoid function for numerical stability
        if (x[i] >= 0)
            x[i] = 1 / (1 + exp(-x[i]));
        else
            x[i] = exp(x[i]) / (1 + exp(x[i]));
    }
}

/**
 * @brief Applies the softmax function to each element of an array in place.
 *
 * The softmax function is defined as:
 *   softmax(x[i]) = exp(x[i]) / Σ(exp(x[j]))
 *
 * To improve numerical stability, it subtracts the maximum value from each input before exponentiation:
 *   softmax(x[i]) = exp(x[i] - MAX_VAL) / Σ(exp(x[j] - MAX_VAL))
 *
 * @param x Pointer to an array of doubles representing input values.
 * @param size The number of elements in the array.
 *
 * @note The function modifies the input array in place.
 * @note OpenMP is used to parallelize the computations for performance.
 */
void softmax(float *x, const int rows, const int cols) {

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        // Find the maximum value in the row for stability
        double MAX_VAL = -INFINITY;
        #pragma omp parallel for reduction(max:MAX_VAL)
        for (int j = 0; j < cols; j++) {
            if (x[i*cols + j] > MAX_VAL) {
                MAX_VAL = x[i*cols + j];
            }
        }

        // Find the sum of exponentials of the array elements
        double sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < cols; j++) {
            x[i*cols + j] = exp(x[i*cols + j] - MAX_VAL);
            sum += x[i*cols + j];
        }

        // Normalize the array by the sum
        #pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            x[i*cols + j] /= sum;
        }
    }
}