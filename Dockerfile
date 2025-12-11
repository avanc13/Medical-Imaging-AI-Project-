# Base image with Python + CUDA + PyTorch runtime
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn9-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set up work directory
WORKDIR /app

# Install system deps (nibabel etc. don't need much)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your repo into the container
COPY . /app

# Install Python dependencies
# If you have a requirements.txt, use that. Otherwise, minimal explicit list:
RUN pip install --no-cache-dir \
    numpy \
    nibabel \
    scipy \
    scikit-image \
    tqdm \
    scikit-learn \
    matplotlib

# Make sure Python can see your package structure
ENV PYTHONPATH=/app

# Default command: do nothing by itself.
# The instructor will always append "python inference/....py ...".
CMD ["bash"]
