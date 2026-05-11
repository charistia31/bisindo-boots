FROM python:3.11-slim

# Install system dependencies termasuk libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements dulu (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Railway inject $PORT, gunicorn baca dari env
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app