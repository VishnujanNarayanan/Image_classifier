"""Gradio demo for the trained age/gender model.

The notebooks train and print; there is nothing to interact with. This wraps the
model in the shape the classifier is actually used in -- drop in a photo, get a
prediction -- which is also what makes the project demonstrable on video.

Predictions go through the HTTP API rather than through a second in-process copy
of the inference code. Point API_URL at a running service to use it; leave it
unset and the demo loads the model itself, which is what the Hugging Face Space
does since it runs as a single process.

    uvicorn scripts.api:app --port 8000 &
    API_URL=http://127.0.0.1:8000 python scripts/demo.py   # http://127.0.0.1:7861
"""
import os
import sys

import cv2
import gradio as gr
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agc import faces, inference  # noqa: E402

API_URL = os.environ.get("API_URL")
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "UTKFace", "images_flat", "part1")

_local = {"model": None, "detector": None}


def _local_model():
    if _local["model"] is None:
        _local["model"] = inference.load_model(os.environ.get("MODEL_PATH"))
        _local["detector"] = faces.detector()
        print("loaded model in-process", flush=True)
    return _local["model"], _local["detector"]


def _via_api(bgr):
    """Ask the running service. Its 422 is a real answer, not a failure."""
    blob = cv2.imencode(".jpg", bgr)[1].tobytes()
    r = httpx.post(f"{API_URL}/predict", files={"file": ("upload.jpg", blob, "image/jpeg")},
                   timeout=30)
    if r.status_code == 422:
        raise inference.NoFaceDetected(r.json()["detail"])
    r.raise_for_status()
    return r.json()


def predict(rgb):
    if rgb is None:
        return "no image", "", None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        if API_URL:
            result = _via_api(bgr)
            # the API returns numbers only, so the crop is recomputed here purely
            # to show the viewer what the model was actually looking at
            crop = faces.align_crop(bgr, _detector_for_display())
        else:
            result, crop = inference.predict(bgr, *_local_model())
    except inference.NoFaceDetected:
        return "no face detected", "", None
    gender = f"{result['gender']}  ({result['gender_confidence']:.0%} confidence)"
    return (f"{result['age']:.0f} years", gender,
            None if crop is None else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def _detector_for_display():
    if _local["detector"] is None:
        _local["detector"] = faces.detector()
    return _local["detector"]


examples = []
if os.path.isdir(SAMPLES):
    files = sorted(os.listdir(SAMPLES))
    for want in (3, 16, 26, 35, 51, 68):        # spread across the age range
        hit = next((f for f in files if f.split("_")[0] == str(want)), None)
        if hit:
            examples.append(os.path.join(SAMPLES, hit))

with gr.Blocks(title="Age & Gender Classifier") as demo:
    gr.Markdown("# Age & Gender Classifier\n"
                "Multi-task CNN over UTKFace. A face is detected and aligned, then one "
                "shared trunk predicts an age distribution and a gender.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="face photo", height=320)
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_age = gr.Textbox(label="Predicted age")
            out_gen = gr.Textbox(label="Predicted gender")
            out_face = gr.Image(label="Detected face (model input)", height=160)
    if examples:
        gr.Examples(examples=examples, inputs=inp, label="Sample faces")
    btn.click(predict, inp, [out_age, out_gen, out_face])
    inp.change(predict, inp, [out_age, out_gen, out_face])

if __name__ == "__main__":
    print(f"predictions via {'API at ' + API_URL if API_URL else 'in-process model'}", flush=True)
    demo.launch(server_name=os.environ.get("HOST", "127.0.0.1"),
                server_port=int(os.environ.get("PORT", 7861)))
