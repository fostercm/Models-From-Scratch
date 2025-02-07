# Use CUDA 12.0 as base image
FROM nvidia/cuda:12.0-devel-ubuntu24.04

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    cmake make g++ gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Build the project
RUN mkdir -p build 
WORKDIR /app/build
RUN cmake .. && make

# Set working directory in container
WORKDIR /app

# Expose port
EXPOSE 8000

# Start the application with Uvicorn
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# # Expose port
# EXPOSE 8000

# # Start the application with Uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]