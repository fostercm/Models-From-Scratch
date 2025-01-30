#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H

void printMatrixCUDA(const float *matrix, const int rows, const int cols);
__global__ void populateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n);
void launchPopulateDiagonalKernel(float *matrix, const float *diagonal, const int m, const int n);

#endif /* MATRIX_FUNCTIONS_H */
