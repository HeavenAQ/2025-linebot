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

ARG REQUIRE_TRT_ENGINE=1
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools==80.10.2 wheel && \
    pip install -r requirements.txt && \
    pip install --no-deps \
      tensorrt-cu12-libs==10.14.1.48.post1 \
      tensorrt-cu12-bindings==10.14.1.48.post1 \
      tensorrt-cu12==10.14.1.48.post1 && \
    find /opt/conda/lib/python3.11/site-packages/tensorrt_libs \
      -type f -name 'libnvinfer_builder_resource_*' \
      ! -name 'libnvinfer_builder_resource_sm89.so.10.14.1' -delete && \
    if [ "$REQUIRE_TRT_ENGINE" = "0" ]; then \
      pip install --no-cache-dir 'rfdetr[onnx]==1.9.1'; \
    fi

RUN test -f /opt/conda/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_builder_resource_sm89.so.10.14.1 && \
    python -c "import torch; assert torch.version.cuda == '12.4'; import tensorrt as trt; assert trt.__version__ == '10.14.1.48.post1'"
COPY . .

RUN python -c "import service.server; print('Python service imports verified')"

# A missing engine is not a runtime error — the detector would just rebuild it
# on every cold start, which is precisely the cost this avoids. So deployable
# images must carry it. The exception is the bootstrap build used to produce
# the engine in the first place, which cannot contain what it exists to create.
RUN if [ "$REQUIRE_TRT_ENGINE" = "1" ]; then \
      test "$(find models/trt-engines -name 'rfdetr-keypoint-preview.trt' | wc -l)" -ge 1 && \
      echo "Prebuilt RF-DETR TensorRT engine included"; \
    else \
      echo "Bootstrap build: RF-DETR engine not required"; \
    fi

EXPOSE 8080
CMD ["python", "-m", "service.server"]
