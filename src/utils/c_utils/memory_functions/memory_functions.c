#include "memory_functions.h"
#include <stdio.h>
#include <stdlib.h>

// Function to allocate memory for a matrix
void* safeMalloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Function to free memory
void safeFree(void* d_ptr) {
    if (d_ptr != NULL) {
        free(d_ptr);
    }
}