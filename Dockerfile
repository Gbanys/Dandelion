# Dockerfile
FROM python:3.10-slim

# Install system deps including libGL for OpenCV
RUN apt-get update && \
    apt-get install -y \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir fastapi uvicorn pillow ultralytics python-multipart

# Copy your app
WORKDIR /app
COPY . .

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

