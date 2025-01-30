#include "pseudoinverse.h"
#include "../c_memory_functions/memory_functions.h"
#include <lapacke.h>
#include <cblas.h>
#include <stdio.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Function to compute the pseudoinverse of a matrix
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