#include <stdio.h>
#include "c_utils.h"

// Function to print a matrix
void printMatrix(const float *A, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}