# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(PCACUDABackend LANGUAGES CUDA)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Set architecture for the GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/MLFromScratch/utils/cuda_utils)

# Source files
set(SOURCES 
    principal_component_analysis.cu
    )

# Create the shared library
add_library(principal_component_analysis_cuda SHARED ${SOURCES})

# Link the shared library with the required libraries
target_link_libraries(principal_component_analysis_cuda PRIVATE
    cuda_utils
    cudart
    cublas
    cusolver
)