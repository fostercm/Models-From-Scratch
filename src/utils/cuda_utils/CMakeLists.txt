# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(cuda_utils LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Source files for the library
set(SOURCES pseudoinverse.cpp memory_functions.cu loss_functions.cu loss_functions.cpp)

# Set architecture for the GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Create the shared library
add_library(cuda_utils STATIC ${SOURCES})

# -fPIC is needed for the shared library
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Link the shared library with the required libraries
target_link_libraries(cuda_utils PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)