# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(LinearRegressionCBackend)

# Find packages
find_package(GSL REQUIRED)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
        ${GSL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/utils/c_utils
)

# Source files
set(SOURCES linear_regression.c)

# Create the shared library
add_library(linear_regression_c SHARED ${SOURCES})

# Link libraries
find_library (LAPACKE lapacke)
target_link_libraries(linear_regression_c 
        c_utils
        ${GSL_LIBRARIES}
        ${LAPACK_LIBRARIES}
        -llapacke
)