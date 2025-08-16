# Use a NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Define the default command to run the CIFAR-10 training script
CMD ["python3.10", "src/train_cifar10.py"]