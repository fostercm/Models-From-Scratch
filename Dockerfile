# Use CUDA 12.0 as base image
FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv git \
    libgsl-dev liblapacke-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Ensure the virtual environment is used by default
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy API folder
COPY api ./api

# Set working directory in container
WORKDIR /app

# Expose port
EXPOSE 8000

# Start the application with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

## Docker commands
# docker build -t ml-from-scratch-backend .
# docker run --rm --gpus all -p 8000:8000 ml-from-scratch-backend:latest