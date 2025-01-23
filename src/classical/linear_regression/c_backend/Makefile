# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC
LDFLAGS = -shared

# Source file and output
SRC = linear_regression.c
TARGET = linear_regression.so

# Default target
all: $(TARGET)

# Create the shared library directly from the source file
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Clean build files
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean