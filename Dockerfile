# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies if required for model/compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy our ultra-optimized backend script
COPY rag_api.py .



# Expose port (Render automatically provides PORT env variable, defaulting to 8000 here)
EXPOSE 8000

# Start Uvicorn server, using the PORT environment variable that cloud hosts inject
CMD ["sh", "-c", "uvicorn rag_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
