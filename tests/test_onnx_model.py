"""End-to-end tests against the REAL exported model.

Everything else in this suite stubs the model out, which is right for checking
routing and error paths but proves nothing about the network actually shipped.
These run the committed ONNX graph. It is 1.4MB, so it lives in the repo and CI
can exercise it -- the Keras original at 4.2MB plus a TensorFlow install could
not have been checked here at all.
"""
import os

import numpy as np
import pytest

from agc.labels import NUM_BINS, age_grid

pytest.importorskip("onnxruntime", reason="serving runtime")

from agc.inference import (  # noqa: E402
    GENDERS,
    NoFaceDetected,
    OnnxModel,
    load_model,
    predict,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "artifacts", "deep.onnx")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL), reason="artifacts/deep.onnx not exported yet")


class _Detector:
    """Returns the middle half of whatever it is given."""

    def __init__(self, found=True):
        self.found = found

    def process(self, _rgb):
        box = type("B", (), {"xmin": 0.25, "ymin": 0.25, "width": 0.5, "height": 0.5})()
        det = type("D", (), {"location_data": type("L", (), {"relative_bounding_box": box})()})()
        return type("R", (), {"detections": [det] if self.found else []})()


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL)


def test_an_onnx_path_selects_the_onnx_runtime(model):
    assert isinstance(model, OnnxModel)


def test_the_heads_have_the_shapes_the_rest_of_the_code_assumes(model):
    out = model.predict(np.zeros((1, 64, 64, 3), dtype="float32"))
    age, gender = np.asarray(out[0]), np.asarray(out[1])
    assert age.shape == (1, NUM_BINS)
    assert gender.shape == (1, 2)


def test_both_heads_are_probability_distributions(model):
    rng = np.random.default_rng(0)
    out = model.predict(rng.random((4, 64, 64, 3)).astype("float32"))
    for head in out:
        head = np.asarray(head)
        assert np.allclose(head.sum(axis=1), 1.0, atol=1e-5)
        assert (head >= 0).all()


def test_predicted_ages_land_inside_the_grid(model):
    rng = np.random.default_rng(1)
    dist = np.asarray(model.predict(rng.random((16, 64, 64, 3)).astype("float32"))[0])
    ages = dist @ age_grid()
    assert ((ages >= 0) & (ages <= NUM_BINS - 1)).all()


def test_a_full_prediction_has_the_documented_shape(model):
    img = (np.random.default_rng(2).random((200, 200, 3)) * 255).astype(np.uint8)
    result, crop = predict(img, model, _Detector())
    assert set(result) == {"age", "gender", "gender_confidence"}
    assert 0 <= result["age"] <= 100
    assert result["gender"] in GENDERS
    assert 0.5 <= result["gender_confidence"] <= 1.0   # it is the argmax of two
    assert crop.shape == (64, 64, 3)


def test_no_face_still_raises_with_the_real_model(model):
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(NoFaceDetected):
        predict(img, model, _Detector(found=False))


def test_the_same_face_gives_the_same_answer_twice(model):
    """Deterministic inference -- no dropout or augmentation leaking into serving."""
    img = (np.random.default_rng(3).random((150, 150, 3)) * 255).astype(np.uint8)
    first, _ = predict(img, model, _Detector())
    second, _ = predict(img, model, _Detector())
    assert first == second
