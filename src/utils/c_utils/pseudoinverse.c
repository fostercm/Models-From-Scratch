#include "pseudoinverse.h"
#include <lapacke.h>
#include <cblas.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Function to compute the pseudoinverse of a matrix
void computePseudoinverse(float *A, float *A_inv, int m, int n) {

    // Allocate memory for the SVD results
    float *U = (float *) malloc(m * m * sizeof(float));
    float *S = (float *) malloc(MIN(m, n) * sizeof(float));
    float *V_transpose = (float *) malloc(n * n * sizeof(float));
    float *superb = (float *) malloc(MIN(m, n) * sizeof(float));

    // Allocate memory for matrix multiplication intermediates
    float *S_inverse = (float *) calloc(m * n, sizeof(float));
    float *VS_inverse = (float *) malloc(n * m * sizeof(float));
    
    // Check if memory allocation failed
    if (U == NULL || S == NULL || V_transpose == NULL || superb == NULL || S_inverse == NULL || VS_inverse == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Compute the SVD and free superb
    int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, n, S, U, m, V_transpose, n, superb);
    free(superb);

    // Check if the SVD was successful
    if (info > 0) {
        fprintf(stderr, "The algorithm computing SVD failed to converge.\n");
        exit(EXIT_FAILURE);
    }

    // Diagonalize and invert non-zero singular values (create S+)
    for (int i = 0; i < MIN(m, n); i++) {
        if (S[i] > 1e-15) {
            S_inverse[i * m + i] = 1.0 / S[i];
        }
    }

    // Free S
    free(S);

    // Multiply V (V_transpose_transpose) and S+ to get VS_inverse
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, n, 1, V_transpose, n, S_inverse, m, 0, VS_inverse, m);

    // Free VT and S_Inverse
    free(V_transpose);
    free(S_inverse);

    // Multiply VS_inverse and U_transpose to get the pseudoinverse
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, m, 1, VS_inverse, m, U, m, 0, A_inv, m);

    // Free U and VSI
    free(U);
    free(VS_inverse);
}