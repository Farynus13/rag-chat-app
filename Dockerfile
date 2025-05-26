# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into container
COPY . .

# Install only necessary Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    sentence-transformers \
    faiss-cpu

# Expose FastAPI port
EXPOSE 10000

# Run the API server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
