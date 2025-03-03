#include "principal_component_analysis.h"
#include "c_matrix_functions/matrix_functions.h"
#include "c_memory_functions/memory_functions.h"
#include <lapacke.h>
#include <cblas.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int transform(float *X, float *X_transformed, const int n_samples, const int n_features, const int n_components, const float explained_variance_ratio) {
    // Get the number of components to keep
    int n_comp = (n_components > 0) ? n_components : MIN(n_samples, n_features);

    // Allocate memory for the SVD results
    float *U, *S, *V_T;
    U = (float*) safeMalloc(n_samples * n_samples * sizeof(float));
    S = (float*) safeMalloc(MIN(n_samples,n_features) * sizeof(float));
    V_T = (float*) safeMalloc(n_features * n_features * sizeof(float));

    // Standardize the data
    standardize(X, n_samples, n_features);

    // Copy the input matrix to the transformed matrix
    safeMemcpy(X_transformed, X, n_samples * n_features * sizeof(float));

    // Perform SVD
    LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', n_samples, n_features, X_transformed, n_features, S, U, n_samples, V_T, n_features);

    // If explained_variance is specified, use the components that explain the variance
    if (explained_variance_ratio != 0) {
        // Calculate the explained variance
        float total_variance = 0;
        for (int i = 0; i < n_comp; i++) {
            S[i] = S[i] * S[i];
            total_variance += S[i];
        }
        
        // Calculate the cumulative explained variance
        float cumulative_explained_variance = 0;
        for (int i=0; i<n_comp; i++) {
            cumulative_explained_variance += S[i] / total_variance;
            if (cumulative_explained_variance >= explained_variance_ratio) {
                n_comp = i + 1;
                break;
            }
        }
    }

    // Reduce the number of components
    float *V_T_reduced = (float*) safeMalloc(n_comp * n_features * sizeof(float));
    safeMemcpy(V_T_reduced, V_T, n_comp * n_features * sizeof(float));

    // Multiply the input matrix by V matrix
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        n_samples, n_comp, n_features, 
        1.0, 
        X, n_features,
        V_T, n_features, 
        0.0, 
        X_transformed, n_comp
    );

    // Free memory
    safeFree(U);
    safeFree(S);
    safeFree(V_T);

    // Return the number of components
    return n_comp;
}