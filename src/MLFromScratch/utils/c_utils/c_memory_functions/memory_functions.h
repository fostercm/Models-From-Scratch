#ifndef MEMORY_FUNCTIONS_H
#define MEMORY_FUNCTIONS_H

#include <stdlib.h>

void* safeMalloc(size_t size);
void safeFree(void* d_ptr);
void safeMemcpy(void* dest, const void* src, size_t size);

#endif /* MEMORY_FUNCTIONS_H */
