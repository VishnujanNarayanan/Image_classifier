# The demo needed a 5.7GB virtualenv, a specific TensorFlow build and a MediaPipe
# wheel that does not install cleanly everywhere. That is a lot to ask of someone
# who wants to look at the thing for thirty seconds.
FROM python:3.11-slim

# OpenCV needs these even in the headless build; MediaPipe needs libGL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download TensorFlow.
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY agc/ ./agc/
COPY scripts/ ./scripts/
COPY sql/ ./sql/
COPY artifacts/deep.keras ./artifacts/deep.keras

ENV HOST=0.0.0.0 PORT=7861 API_URL=http://127.0.0.1:8000
EXPOSE 7861 8000

# The API is the thing that holds the model; the demo is a client of it.
CMD ["sh", "-c", "uvicorn scripts.api:app --host 0.0.0.0 --port 8000 & exec python scripts/demo.py"]
