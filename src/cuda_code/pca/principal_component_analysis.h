#ifndef PRINCIPAL_COMPONENT_ANALYSIS_H
#define PRINCIPAL_COMPONENT_ANALYSIS_H

#include <cuda_runtime.h>

extern "C" {
    int transform(float *X, float *X_transformed, const int n_samples, const int n_features, const int n_components, const float explained_variance_ratio);
    __global__ void _computeVarianceKernel(const float *d_S, float *d_variance, const int n_features);
}

#endif /* PRINCIPAL_COMPONENT_ANALYSIS_H */
