# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(cuda_utils LANGUAGES CUDA)

# Set architecture for the GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Source files
file(GLOB_RECURSE SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

# Set the flags for the static library
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -O3 -Wall")

# Create the static library
add_library(cuda_utils STATIC ${SOURCES})

# Create the shared library
add_library(cuda_utils_shared SHARED ${SOURCES})

# Link libraries
target_link_libraries(cuda_utils PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

# Link libraries
target_link_libraries(cuda_utils_shared PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)