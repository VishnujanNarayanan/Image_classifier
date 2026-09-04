"""HTTP API over the trained age/gender model, and the page that uses it.

The Gradio demo was the only way to get a prediction, which meant the model could
not be called by anything that was not a person with a browser. This exposes the
same inference behind POST /predict, with a small static page at / for people.

Gradio is deliberately not in this path. It cost 1.6s of import and ~150MB of
resident memory, both of which matter on a free tier that spins down after 15
minutes and wakes on a 0.1-CPU instance. scripts/demo.py still uses it for local
work; nothing deployed does.

    uvicorn scripts.api:app --host 0.0.0.0 --port 8000
"""
import base64
import os
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agc import inference  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(HERE, "ui.html")

#: Populated at startup, or on first use if eager loading was skipped. Tests
#: patch this directly to inject a stub.
_state = {"model": None, "detector": None}


def get_model():
    if _state["model"] is None:
        _state["model"], _state["detector"] = inference.shared_model_and_detector(
            os.environ.get("MODEL_PATH"))
    return _state["model"], _state["detector"]


@asynccontextmanager
async def lifespan(_app):
    """Load the model before the port starts serving.

    Lazily loading it instead would push ~0.8s of work onto whoever sends the
    first request. Doing it here means the keep-warm ping absorbs that cost and a
    real visitor never sees it. EAGER_LOAD=0 restores the lazy behaviour for
    tests and for anyone without the model file.
    """
    if os.environ.get("EAGER_LOAD", "1") != "0":
        try:
            get_model()
            print("model loaded at startup", flush=True)
        except Exception as exc:                     # pragma: no cover
            # A missing model must not stop the app booting -- /health should be
            # able to report the problem rather than the process dying silently.
            print(f"eager load failed ({type(exc).__name__}: {exc})", flush=True)
    yield


app = FastAPI(title="Age & Gender Classifier",
              description="Predicts age and gender from a single face photo.",
              version="1.0.0", lifespan=lifespan)


class Prediction(BaseModel):
    age: float
    gender: str
    gender_confidence: float
    crop_png_base64: str | None = None
    #: The full 101-bin age distribution. This model does not predict a number,
    #: it predicts a shape, and the reported age is that shape's expectation --
    #: so a caller that wants to show how confident the answer is needs the bins.
    age_distribution: list[float] | None = None


class Health(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", model_loaded=_state["model"] is not None)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI)


@app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
async def predict(file: UploadFile = File(...), crop: bool = False, dist: bool = False):
    """Predict age and gender.

    `crop=true` also returns the face the model saw; `dist=true` returns the full
    101-bin age distribution behind the single number.
    """
    raw = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode that file as an image")
    try:
        result, face = inference.predict(image, *get_model(), want_distribution=dist)
    except inference.NoFaceDetected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if crop:
        result["crop_png_base64"] = base64.b64encode(
            cv2.imencode(".png", face)[1].tobytes()).decode()
    return Prediction(**result)
