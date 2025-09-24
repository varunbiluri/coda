# Dockerfile for Coda testing sandbox
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies that are commonly needed
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pytest>=7.4.0 \
    httpx>=0.25.0 \
    pydantic>=2.5.0

# Default command
CMD ["pytest", "-q"]
