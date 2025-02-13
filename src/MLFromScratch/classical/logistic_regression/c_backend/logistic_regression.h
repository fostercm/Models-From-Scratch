#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

void fit(float *X, float *Y, int n_samples, int n_features, float *weights, int max_iter, float lr);
void predict(const float *X, const float *Beta, float *Prediction, const int n_samples, const int n_input_features, const int n_classes);
float cost(const float *Y_pred, const float *Y, const int n_samples, const int n_classes);

#endif /* LOGISTIC_REGRESSION_H */
