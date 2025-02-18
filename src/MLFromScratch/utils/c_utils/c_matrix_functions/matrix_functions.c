/**
 * @file matrix_functions.c
 * @brief Provides utility functions for matrix operations.
 *
 * This file contains functions for common matrix operations, 
 * such as printing matrices in a readable format.
 */

#include <stdio.h>
#include "matrix_functions.h"
#include <math.h>

/**
 * @brief Prints a matrix to the standard output.
 *
 * This function prints a matrix stored in row-major order, where 
 * the elements are accessed using the formula A[i * n + j].
 *
 * Example output for a 2x3 matrix:
 * ```
 * 1.000000 2.000000 3.000000
 * 4.000000 5.000000 6.000000
 * ```
 *
 * @param[in] A Pointer to the matrix stored as a 1D array.
 * @param[in] m Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 */
void printMatrix(const float *A, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void standardize(float *X, const int m, const int n) {
    // Define variables for mean and standard deviation
    float mean, std_dev;

    for (int j=0; j<n; j++) {
        mean = 0.0;
        std_dev = 0.0;

        // Calculate mean
        for (int i=0; i<m; i++) {
            mean += X[i * n + j];
        }
        mean /= m;

        // Calculate standard deviation
        for (int i=0; i<m; i++) {
            std_dev += (X[i * n + j] - mean) * (X[i * n + j] - mean);
        }
        std_dev = sqrt(std_dev / (m-1));

        // Standardize the column
        for (int i=0; i<m; i++) {
            X[i * n + j] = (X[i * n + j] - mean) / std_dev;
        }
    }
}