FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/generated \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/tensorrt_libs:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-pose.txt ./
RUN pip install --upgrade pip setuptools==80.10.2 wheel && \
    pip install -r requirements.txt && \
    pip install --no-deps -r requirements-pose.txt

RUN python -c "from pathlib import Path; from rtmlib.tools.file import download_checkpoint; root=Path('/app/models/pose'); root.mkdir(parents=True, exist_ok=True); download_checkpoint('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', dst_dir=str(root), progress=False); download_checkpoint('https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx', dst_dir=str(root), progress=False); print('RTMW3D ONNX models cached')"

COPY . .

RUN test "$(find models/tensorrt-cache/detector -name '*.engine' | wc -l)" -ge 1 && \
    test "$(find models/tensorrt-cache/pose -name '*.engine' | wc -l)" -ge 1 && \
    test "$(find models/tensorrt-cache/correctors -name '*.engine' | wc -l)" -ge 4 && \
    echo "Prebuilt TensorRT detector, pose, and correction engines included"

EXPOSE 8080
CMD ["python", "-m", "service.server"]
