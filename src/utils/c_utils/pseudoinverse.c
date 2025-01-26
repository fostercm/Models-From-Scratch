#include "pseudoinverse.h"
#include <lapacke.h>
#include <cblas.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void print_matrix(double *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void compute_pseudoinverse(double *A, double *A_inv, int m, int n) {

    // Allocate memory for the SVD results
    double *U = (double *) malloc(m * m * sizeof(double));
    double *S = (double *) malloc(MIN(m, n) * sizeof(double));
    double *VT = (double *) malloc(n * n * sizeof(double));
    double *superb = (double *) malloc(MIN(m, n) * sizeof(double));

    // Allocate memory for matrix multiplication intermediates
    double *SI = (double *) calloc(m * n, sizeof(double));
    double *VSI = (double *) malloc(n * m * sizeof(double));
    
    if (U == NULL || S == NULL || VT == NULL || superb == NULL || SI == NULL || VSI == NULL) {
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Compute the SVD and free superb
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, n, S, U, m, VT, n, superb);
    free(superb);

    // Check if the SVD was successful
    if (info > 0) {
        printf("The algorithm computing SVD failed to converge.\n");
        exit(EXIT_FAILURE);
    }

    // Diagonalize and invert non-zero singular values (create S+)
    for (int i = 0; i < MIN(m, n); i++) {
        if (S[i] > 1e-15) {
            SI[i * m + i] = 1.0 / S[i];
        }
    }

    // Free S
    free(S);

    // Multiply V (VT_T) and S+ to get VSI
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, n, 1, VT, n, SI, m, 0, VSI, m);

    // Free VT and SI
    free(VT);
    free(SI);

    // Multiply VSI and U^T to get the pseudoinverse (A_inv = VSI * U_T)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, m, 1, VSI, m, U, m, 0, A_inv, m);

    // Free U and VSI
    free(U);
    free(VSI);
}