# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(cuda_utils LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set the include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mathematical_functions
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_memory_functions
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_loss_functions
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_matrix_functions
)

# Source files for the library
file(GLOB_RECURSE SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

# Set architecture for the GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Create the static library
add_library(cuda_utils STATIC ${SOURCES})

# -fPIC is needed for the shared library
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Link the shared library with the required libraries
target_link_libraries(cuda_utils PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)