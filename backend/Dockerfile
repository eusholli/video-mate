FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# ffmpeg is required for video processing
# git is required for pip installing from git
# build-essential (gcc) is often needed for compiling python extensions
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install standard dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ImageBind separately with the flag as requested
RUN pip install --no-deps "git+https://github.com/facebookresearch/ImageBind.git@3fcf5c9039de97f6ff5528ee4a9dce903c5979b3"

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 64451

# Set environment variables
# PYTHONUNBUFFERED=1 ensures logs are sent directly to container logs
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "videorag_api.py"]
