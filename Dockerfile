# Use CUDA 12.0 as base image
FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Ensure the virtual environment is used by default
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy API folder
COPY app/api ./app/api

# Copy precompiled C/C++/CUDA libraries to the container
COPY build/lib /app/build/lib

# Set working directory in container
WORKDIR /app

# Expose port
EXPOSE 8000

# Start the application with Uvicorn
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]