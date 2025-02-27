#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

float meanSquaredError(const float *Y_pred, const float *Y, const int num_samples, const int num_output_features);
float crossEntropy(const float *Y_pred, const float *Y, const int num_samples, const int num_classes);

#endif /* LOSS_FUNCTIONS_H */
