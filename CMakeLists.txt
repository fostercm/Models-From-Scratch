cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Models-From-Scratch VERSION 1.0)

# Set the output directories for the compiled library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add the static library subdirectories to the build process
add_subdirectory(src/utils/c_utils)
add_subdirectory(src/utils/cuda_utils)

# Add shared library subdirectories to the build process
add_subdirectory(src/classical/linear_regression/c_backend)
add_subdirectory(src/classical/linear_regression/cuda_backend)