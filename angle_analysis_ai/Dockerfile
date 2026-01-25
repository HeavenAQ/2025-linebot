FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System libs needed by OpenCV/mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO weights at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"

COPY . .

EXPOSE 8080
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8080"]

