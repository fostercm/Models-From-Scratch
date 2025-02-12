/**
 * @file memory_functions.c
 * @brief Provides safe memory allocation and deallocation functions.
 *
 * This file contains helper functions for dynamic memory management, 
 * ensuring proper allocation and error handling.
 */

#include "memory_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Allocates memory safely and exits on failure.
 *
 * This function attempts to allocate the requested memory using `malloc()`. 
 * If allocation fails, it prints an error message and terminates the program.
 *
 * @param[in] size Number of bytes to allocate.
 * @return Pointer to the allocated memory.
 */
void* safeMalloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

/**
 * @brief Safely frees allocated memory.
 *
 * This function checks if the pointer is non-null before calling `free()`, 
 * preventing potential double-free errors.
 *
 * @param[in] d_ptr Pointer to the allocated memory to be freed.
 */
void safeFree(void* d_ptr) {
    if (d_ptr != NULL) {
        free(d_ptr);
    }
}

/**
 * @brief Safely copies memory from source to destination.
 *
 * This function checks if the source and destination pointers are non-null 
 * before calling `memcpy()`, preventing potential segmentation faults.
 *
 * @param[out] dest Pointer to the destination memory.
 * @param[in] src Pointer to the source memory.
 * @param[in] size Number of bytes to copy.
 */
void safeMemcpy(void* dest, const void* src, size_t size) {
    if (dest == NULL || src == NULL) {
        fprintf(stderr, "Error: Invalid memory copy operation\n");
        exit(EXIT_FAILURE);
    }
    memcpy(dest, src, size);
}