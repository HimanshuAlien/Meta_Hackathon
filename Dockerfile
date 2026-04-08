# Official Python Image
FROM python:3.11-slim-bookworm




# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 5 -r requirements.txt


# Copy the rest of the application
COPY . .

# Remove any stale bytecode that might override source files
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
RUN find /app -name "*.pyc" -delete 2>/dev/null; true

# Expose port (OpenEnv standard for HF Spaces)
EXPOSE 7860

# Environment variables (can be overridden by HF Spaces Secrets)
ENV MODEL_NAME="distilgpt2"
ENV HF_TOKEN=""

# Run the OpenEnv API server by default
CMD ["python", "server/app.py"]
