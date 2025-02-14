#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

void fit(const float *X, const float *Y, float *Beta, const int n_samples, const int n_input_features, const int n_classes, const int max_iters, const float lr);
void predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes);
float cost(const float *Y_pred, const float *Y, const int n_samples, const int n_classes);

#endif /* LOGISTIC_REGRESSION_H */
