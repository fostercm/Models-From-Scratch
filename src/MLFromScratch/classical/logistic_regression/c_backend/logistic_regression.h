#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

void fit(double *X, double *Y, int n_samples, int n_features, double *weights, int max_iter, double lr);
void predict(double *X, double *weights, int n_samples, int n_features, double *preds);
float cost(double *Y_pred, double *Y, int n_samples, int n_classes);

#endif /* LOGISTIC_REGRESSION_H */
