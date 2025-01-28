#include <cuda_runtime.h>
#include "linear_regression_cuda.h"

__global__ void kernel_cost() {
    printf("Hello from kernel_cost\n");
}