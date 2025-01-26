# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(LinearRegressionCBackend)

# Find the GSL package
find_package(GSL REQUIRED)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${LAPACK_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src/utils/c_utils)

# Source files for the library
set(SOURCES linear_regression.c)

# Create the shared library
add_library(linear_regression_c SHARED ${SOURCES})

# Set the output directory for the shared library
set_target_properties(linear_regression_c PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Link the library to GSL, LAPACK, and c_utils
find_library (LAPACKE lapacke)
target_link_libraries(linear_regression_c 
        c_utils
        ${GSL_LIBRARIES}
        ${LAPACK_LIBRARIES}
        -llapacke
)