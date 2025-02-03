/**
 * @file inverse.c
 * @brief Computes the inverse of a square matrix, with fallback to pseudoinverse.
 *
 * This file provides a function to compute the inverse of a given square matrix 
 * using LU decomposition. If the matrix is singular, the function falls back 
 * to computing the Moore-Penrose pseudoinverse.
 *
 * The implementation utilizes LAPACKE for efficient matrix operations 
 * and relies on safe memory allocation functions to prevent memory errors.
 */


#include "inverse.h"
#include "pseudoinverse.h"
#include "../c_memory_functions/memory_functions.h"
#include "../c_matrix_functions/matrix_functions.h"
#include <lapacke.h>

/**
 * @brief Computes the inverse of a square matrix, with fallback to pseudoinverse.
 *
 * This function computes the inverse of an `n x n` matrix using LU decomposition. 
 * If the matrix is singular (i.e., non-invertible), the function instead computes 
 * the Moore-Penrose pseudoinverse. The operation is performed in-place on the 
 * provided output matrix.
 *
 * @param[in] A Pointer to the input square matrix of size `n x n`.
 * @param[out] A_inv Pointer to the output matrix where the inverse or pseudoinverse is stored.
 * @param[in] n The size (number of rows and columns) of the square matrix.
 */

void computeInverse(float *A, float *A_inv, const int n) {
    // Allocate memory for the LU decomposition
    int info;
    int *ipiv;
    ipiv = (int*) safeMalloc(n * sizeof(int));

    // Copy the input matrix
    safeMemcpy(A_inv, A, n * n * sizeof(float));

    // Perform LU decomposition
    info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, n, n, A_inv, n, ipiv);

    // Check if the matrix is singular
    if (info != 0) {
        // Compute the pseudoinverse
        computePseudoinverse(A, A_inv, n, n);
    }
    else {
        // Compute the inverse
        LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, A_inv, n, ipiv);
    }

    // Free memory
    safeFree(ipiv);
}