# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(PCACBackend)

# Find packages
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
        ${BLAS_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/MLFromScratch/utils/c_utils
)

# Source files
set(SOURCES principal_component_analysis.c)

# Set the flags for the static library
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fopenmp -O3 -march=native -Wall")

# Create the shared library
add_library(principal_component_analysis_c SHARED ${SOURCES})

# Link libraries
target_link_libraries(principal_component_analysis_c 
        c_utils
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        -llapacke
)