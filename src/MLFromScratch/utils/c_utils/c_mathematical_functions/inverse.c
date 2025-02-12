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
#include "../c_memory_functions/memory_functions.h"
#include "../c_matrix_functions/matrix_functions.h"
#include <lapacke.h>
#include <cblas.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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

/**
 * @brief Computes the Moore-Penrose pseudoinverse of a matrix.
 *
 * This function performs singular value decomposition (SVD) on the input 
 * matrix A and calculates its pseudoinverse A+ as:
 * A+ = V S+ U^T
 * 
 * where:
 *  - U is an m x m orthogonal matrix,
 *  - S is an m x n diagonal matrix of singular values,
 *  - V is an n x n orthogonal matrix,
 *  - S+ is the inverse of nonzero singular values in S
 *
 * @param[in]  A   Pointer to the input matrix of size m x n.
 * @param[out] A_inv Pointer to the output pseudoinverse matrix of size n x m.
 * @param[in]  m   Number of rows in the input matrix.
 * @param[in]  n   Number of columns in the input matrix.
 */
void computePseudoinverse(float *A, float *A_inv, const int m, const int n) {

    // Allocate memory for the SVD results
    float *U, *S, *V_T;
    U = (float*) safeMalloc(m * m * sizeof(float));
    S = (float*) safeMalloc(MIN(m, n) * sizeof(float));
    V_T = (float*) safeMalloc(n * n * sizeof(float));

    // Compute the SVD
    LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, A, n, S, U, m, V_T, n);

    // Diagonalize and invert non-zero singular values (create S+)
    float *S_inv;
    S_inv = (float*) safeMalloc(m * n * sizeof(float));
    memset(S_inv, 0, m * n * sizeof(float));
    for (int i = 0; i < MIN(m, n); i++) {
        if (S[i] > 1e-15) {
            S_inv[i * m + i] = 1.0 / S[i];
        }
    }

    // Free S
    safeFree(S);

    // Compute VS_inv = V * S_inv
    float *VS_inv;
    VS_inv = (float *) safeMalloc(n * m * sizeof(float));
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans, 
        n, m, n, 
        1, 
        V_T, n, 
        S_inv, m, 
        0, 
        VS_inv, m);

    // Free VT and S_Inverse
    safeFree(V_T);
    safeFree(S_inv);

    // Compute A_inv = VS_inv * U (V * S+ * U)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        n, m, m, 
        1, 
        VS_inv, m, 
        U, m, 
        0, 
        A_inv, m);

    // Free U and VSI
    safeFree(U);
    safeFree(VS_inv);
}