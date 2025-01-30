/**
 * @file pseudoinverse.c
 * @brief Computes the Moore-Penrose pseudoinverse of a matrix using SVD.
 *
 * This implementation utilizes LAPACK and CBLAS for efficient singular 
 * value decomposition (SVD) to compute the pseudoinverse. The function 
 * follows the standard approach: 
 * A+ = V S+ U^T
 * where S+ is the diagonal matrix of inverted nonzero singular values.
 *
 * @author  Cole Foster
 * @date    2025-01-29
 */

#include "pseudoinverse.h"
#include "../c_memory_functions/memory_functions.h"
#include <lapacke.h>
#include <cblas.h>
#include <stdio.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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
    float *U, *S, *V_T, *superb;
    U = (float*) safeMalloc(m * m * sizeof(float));
    S = (float*) safeMalloc(MIN(m, n) * sizeof(float));
    V_T = (float*) safeMalloc(n * n * sizeof(float));
    superb = (float*) safeMalloc(MIN(m, n) * sizeof(float));

    // Compute the SVD and free superb
    LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, n, S, U, m, V_T, n, superb);
    safeFree(superb);

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
    VS_inv = (float *) malloc(n * m * sizeof(float));
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