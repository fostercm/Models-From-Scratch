# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(c_utils)

# Find LAPACK if it's needed for utils
find_package(LAPACK REQUIRED)

# Source files for the library
set(SOURCES pseudoinverse.c)

# Create the static library
add_library(c_utils STATIC ${SOURCES})

# Set the output directory for the static library
set_target_properties(c_utils PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Link LAPACK to the utils library (use LAPACK_LIBRARIES)
target_link_libraries(c_utils ${LAPACK_LIBRARIES})