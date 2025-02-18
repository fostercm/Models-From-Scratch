#ifndef PRINCIPAL_COMPONENT_ANALYSIS_H
#define PRINCIPAL_COMPONENT_ANALYSIS_H

int transform(float *X, float *X_transformed, const int n_samples, const int n_features, const int N_components, const float explained_variance_ratio);

#endif /* PRINCIPAL_COMPONENT_ANALYSIS_H */
