# Trying the demo previously meant a 5.7GB virtualenv, a specific TensorFlow
# build and a MediaPipe wheel that does not install cleanly everywhere.
#
# Serving runs the ONNX export rather than Keras, so TensorFlow is not installed
# here at all: ~67MB resident instead of ~1GB, which is what makes this fit a
# 512MB free tier. See scripts/export_onnx.py for the conversion and its
# parity check against the original.
FROM python:3.11-slim

# OpenCV and MediaPipe still want these even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the wheels.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY agc/ ./agc/
COPY scripts/ ./scripts/
COPY artifacts/deep.onnx ./artifacts/deep.onnx

ENV HOST=0.0.0.0 PORT=7861 API_URL=http://127.0.0.1:8000
EXPOSE 7861 8000

# The API holds the model; the demo is a client of it.
CMD ["sh", "-c", "uvicorn scripts.api:app --host 0.0.0.0 --port 8000 & exec python scripts/demo.py"]
