FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System libs needed by OpenCV/mediapipe (fixes libgthread-2.0.so.0)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Avoid cache bloat; ensure pip is recent
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache -r requirements.txt

# Copy source
COPY . .

# Cloud Run sets PORT; bind to 0.0.0.0 and respect $PORT
EXPOSE $PORT
CMD ["fastapi", "run", "main.py", "--port", "8080"]
