#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

void fit(double *X, double *Y, double *Beta, int num_samples, int num_input_features, int num_output_features);
void predict(double *X, double *Beta, double *Prediction, int num_samples, int num_input_features, int num_output_features);
double cost(double *Y_pred, double *Y, int num_samples, int num_output_features);

#endif /* LINEAR_REGRESSION_H */