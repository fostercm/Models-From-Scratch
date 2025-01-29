# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(LinearRegressionCUDABackend LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set architecture for the GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/utils/cuda_utils)

# CUDA and C++ source files
set(SOURCES_CUDA linear_regression.cu)
set(SOURCES_CPP linear_regression.cpp)

# Create the shared library
add_library(linear_regression_cuda SHARED ${SOURCES_CUDA} ${SOURCES_CPP})
target_include_directories(linear_regression_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Set the output directory for the shared library
set_target_properties(linear_regression_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Set CUDA-specific compilation flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Link the shared library with the required libraries
target_link_libraries(linear_regression_cuda PRIVATE
    cuda_utils
    cudart
    cublas
    cusolver
)