#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

void fit(float *X, float *Y, float *Beta, int num_samples, int num_input_features, int num_output_features);
void predict(float *X, float *Beta, float *Prediction, int num_samples, int num_input_features, int num_output_features);
float cost(float *Y_pred, float *Y, int num_samples, int num_output_features);

#endif /* LINEAR_REGRESSION_H */