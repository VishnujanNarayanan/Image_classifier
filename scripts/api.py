"""HTTP API over the trained age/gender model.

The Gradio demo was the only way to get a prediction, which meant the model could
not be called by anything that was not a person with a browser. This exposes the
same inference behind POST /predict so the demo, a script, or another service all
go through one path.

    uvicorn scripts.api:app --port 8000
"""
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agc import inference  # noqa: E402

app = FastAPI(title="Age & Gender Classifier",
              description="Predicts age and gender from a single face photo.",
              version="1.0.0")

#: Loaded on first use rather than at import. Keeps the module importable -- and
#: the routing testable -- on a machine with no TensorFlow and no saved model.
_state = {"model": None, "detector": None}


def get_model():
    if _state["model"] is None:
        _state["model"], _state["detector"] = inference.shared_model_and_detector(
            os.environ.get("MODEL_PATH"))
    return _state["model"], _state["detector"]


class Prediction(BaseModel):
    age: float
    gender: str
    gender_confidence: float


class Health(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", model_loaded=_state["model"] is not None)


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    raw = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode that file as an image")
    try:
        result, _crop = inference.predict(image, *get_model())
    except inference.NoFaceDetected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Prediction(**result)
