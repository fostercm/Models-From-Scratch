# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(c_utils)

# Find packages
find_package(LAPACK REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Source files
file(GLOB_RECURSE SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
)

# Set the flags for the static library
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fopenmp -O3 -Wall")

# Create the static library
add_library(c_utils STATIC ${SOURCES})

# Link libraries
target_link_libraries(
    c_utils 
    ${LAPACK_LIBRARIES})