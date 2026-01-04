FROM python:3.10-slim

# Install audio system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This line grabs app.py, main.py, chromaGeneration.py, etc. automatically
COPY . .

# Pointing to app.py
# Use Gunicorn with Uvicorn workers
CMD gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
