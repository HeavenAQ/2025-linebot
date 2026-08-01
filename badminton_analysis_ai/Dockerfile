FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/generated \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libgl1 libglib2.0-0 ninja-build \
    fonts-noto-cjk git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools==80.10.2 wheel && \
    pip install -r requirements.txt && \
    pip install addict==2.4.0 chumpy==0.70 json-tricks==3.17.3 \
        matplotlib==3.10.5 munkres==1.1.4 pycocotools==2.0.10 \
        scipy==1.16.1 six==1.17.0 terminaltables==3.1.10 \
        tqdm==4.67.1 xtcocotools==1.14.3 yapf==0.43.0 && \
    pip install --no-deps mmengine==0.10.7 && \
    MAX_JOBS=4 MMCV_WITH_OPS=1 FORCE_CUDA=1 \
        pip install --no-build-isolation --no-deps mmcv==2.2.0 && \
    pip install --no-deps mmdet==3.2.0 mmpose==1.3.2

COPY . .

# Resolve and cache every OpenMMLab checkpoint while building the image. The
# service must not depend on outbound model downloads during a cold start.
RUN python -c "from badminton_analysis.services.pose_detector import PoseDetector; detector = PoseDetector(); detector.model = detector._load_inferencer(detector.model_name); detector.wholebody_model = detector._load_2d_inferencer(detector.wholebody_model_name); print('MMPose checkpoints cached')"

EXPOSE 8080
CMD ["python", "-m", "service.server"]
