# Development Base
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (OpenEnv standard for HF Spaces)
EXPOSE 7860

# Environment variables (can be overridden by HF Spaces Secrets)
ENV MODEL_NAME="distilgpt2"
ENV HF_TOKEN=""

# Run the OpenEnv API server by default
CMD ["python", "api.py"]
