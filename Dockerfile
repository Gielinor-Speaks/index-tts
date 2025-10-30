# IndexTTS v2 API Dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Set working directory
WORKDIR /app

# Install system dependencies (Ubuntu 24.04 comes with Python 3.12)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all application code (needed for building the package)
COPY . .

# Install dependencies with API extras
RUN uv sync --frozen --no-dev --extra api

# Expose the API port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the API server
CMD ["uv", "run", "api.py", "--port", "8001", "--cuda-kernel"]
