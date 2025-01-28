# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(LinearRegressionCUDABackend)

# Find packages

# Include directories
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${LAPACK_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src/utils/cuda_utils)

# Source files for the library
set(SOURCES linear_regression.cpp linear_regression.cu)

# Create the shared library
add_library(linear_regression_cuda SHARED ${SOURCES})

# Set the output directory for the shared library
set_target_properties(linear_regression_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Link the shared library with the required libraries
# target_link_libraries(linear_regression_c 
#         c_utils
#         ${GSL_LIBRARIES}
#         ${LAPACK_LIBRARIES}
#         -llapacke
# )