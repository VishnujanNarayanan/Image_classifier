"""API routing and error handling, with the model stubbed out.

The point of loading the model lazily is exactly this: the endpoint's contract --
status codes, response shape, what happens to a corrupt upload -- can be checked
without TensorFlow, a saved model, or MediaPipe anywhere on the machine.
"""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")

import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agc import inference  # noqa: E402
from scripts import api  # noqa: E402


class _StubModel:
    """Returns a distribution peaked at 40 and a confident 'Female'."""

    def predict(self, batch, verbose=0):
        age = np.zeros((1, 101), dtype="float32")
        age[0, 40] = 1.0
        return {"age": age, "gender": np.array([[0.1, 0.9]], dtype="float32")}


class _StubDetector:
    def __init__(self, found=True):
        self.found = found

    def process(self, _rgb):
        box = type("B", (), {"xmin": 0.25, "ymin": 0.25, "width": 0.5, "height": 0.5})()
        det = type("D", (), {"location_data": type("L", (), {"relative_bounding_box": box})()})()
        return type("R", (), {"detections": [det] if self.found else []})()


@pytest.fixture
def client(monkeypatch):
    # the stub is injected before startup, so the eager loader finds a model
    # already present and does not try to read one off disk
    monkeypatch.setitem(api._state, "model", _StubModel())
    monkeypatch.setitem(api._state, "detector", _StubDetector())
    return TestClient(api.app)


def _jpeg(size=200):
    img = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health_reports_whether_the_model_is_loaded(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True}


def test_predict_returns_the_documented_shape(client):
    r = client.post("/predict", files={"file": ("face.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"age", "gender", "gender_confidence"}
    assert body["age"] == pytest.approx(40.0)
    assert body["gender"] == "Female"
    assert body["gender_confidence"] == pytest.approx(0.9)


def test_a_file_that_is_not_an_image_is_a_400(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"this is not a jpeg", "text/plain")})
    assert r.status_code == 400
    assert "decode" in r.json()["detail"]


def test_a_photo_with_no_face_is_a_422_not_a_crash(client, monkeypatch):
    monkeypatch.setitem(api._state, "detector", _StubDetector(found=False))
    r = client.post("/predict", files={"file": ("empty.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 422
    assert "no face" in r.json()["detail"]


def test_the_page_is_served_at_the_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Age &amp; Gender Classifier" in r.text
    assert "predict?crop=true" in r.text          # the page calls the real endpoint


def test_the_crop_is_returned_only_when_asked_for(client):
    plain = client.post("/predict", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert "crop_png_base64" not in plain.json()

    withcrop = client.post("/predict?crop=true",
                           files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    body = withcrop.json()
    assert body["crop_png_base64"]
    import base64
    decoded = base64.b64decode(body["crop_png_base64"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"   # a real PNG, not a placeholder


def test_predict_also_hands_back_the_crop_the_model_saw():
    img = np.random.default_rng(1).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    result, crop = inference.predict(img, _StubModel(), _StubDetector())
    assert crop.shape == (64, 64, 3)
    assert result["age"] == pytest.approx(40.0)


def test_predict_raises_when_there_is_no_face():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(inference.NoFaceDetected):
        inference.predict(img, _StubModel(), _StubDetector(found=False))


def test_the_age_distribution_is_returned_only_when_asked_for(client):
    plain = client.post("/predict", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert "age_distribution" not in plain.json()

    asked = client.post("/predict?dist=true", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    dist = asked.json()["age_distribution"]
    assert len(dist) == 101
    assert pytest.approx(sum(dist), abs=1e-3) == 1.0


def test_the_reported_age_is_the_expectation_of_the_returned_distribution(client):
    """The page draws the curve and marks the age on it, so they must agree."""
    body = client.post("/predict?dist=true",
                       files={"file": ("f.jpg", _jpeg(), "image/jpeg")}).json()
    expectation = sum(i * p for i, p in enumerate(body["age_distribution"]))
    assert body["age"] == pytest.approx(expectation, abs=0.1)


def test_the_page_asks_for_both_the_crop_and_the_distribution(client):
    """A silent drift here would leave the signature chart permanently blank."""
    assert "predict?crop=true&dist=true" in client.get("/").text
