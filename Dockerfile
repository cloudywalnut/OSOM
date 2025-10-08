# Use Python 3.12.10 slim image
FROM python:3.12.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for OpenCV / Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into container
COPY . .

# Default port (Railway overrides this automatically)
ENV PORT=5000
EXPOSE 5000

# Use shell form so variable substitution works
CMD gunicorn -k eventlet -w 1 app:app --bind 0.0.0.0:${PORT:-5000}