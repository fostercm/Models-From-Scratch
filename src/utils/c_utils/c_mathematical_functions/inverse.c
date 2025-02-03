#include "inverse.h"
#include "pseudoinverse.h"
#include "../c_memory_functions/memory_functions.h"
#include "../c_matrix_functions/matrix_functions.h"
#include <lapacke.h>

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