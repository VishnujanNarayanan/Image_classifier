"""Turning one photo into one prediction.

Both the HTTP API and the Gradio demo need exactly this, and before now each had
its own copy of the crop-normalise-predict-argmax sequence. The model is loaded
lazily so importing this module costs nothing until a prediction is asked for --
which is what lets the API's routing be tested without any runtime installed.

Serving runs the ONNX export, not the Keras file. Importing TensorFlow costs
roughly a gigabyte resident, which does not fit the 512MB a no-card free tier
gives you; ONNX Runtime loads the same graph in a fraction of that. The two were
checked against each other on real faces and agree to ~3e-07 -- see
scripts/export_onnx.py, which refuses to write a model it cannot verify.
"""
import os

import numpy as np

from agc.faces import IMG_SIZE, align_crop
from agc.labels import NUM_BINS, age_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "artifacts", os.environ.get("MODEL", "deep.onnx"))

GENDERS = ("Male", "Female")


class NoFaceDetected(Exception):
    """The detector found nothing to predict on."""


class OnnxModel:
    """An ONNX session behind the same `.predict(batch)` the Keras model offers.

    Keeping the signature identical is deliberate: `predict` below, and every
    test that stubs a model out, stay unchanged whichever runtime is loaded.
    """

    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, batch, verbose=0):
        # returns [age, gender] in graph order, which read_heads unpacks
        return self.session.run(None, {self.input_name: batch.astype("float32")})


def load_model(path=None):
    """Load whichever runtime the file asks for: .onnx to serve, .keras otherwise."""
    path = path or DEFAULT_MODEL
    if str(path).endswith(".onnx"):
        return OnnxModel(path)
    return load_keras(path)


def load_keras(path):
    """The training-side loader. Needs TensorFlow, so the serving path never calls it."""
    import tensorflow as tf

    def emd_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        return tf.reduce_mean(tf.abs(tf.cumsum(y_true, 1) - tf.cumsum(y_pred, 1)))

    def kl_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

    def age_mae(y_true, y_pred):
        g = tf.constant(age_grid())
        return tf.reduce_mean(tf.abs(tf.tensordot(y_true, g, [[1], [0]])
                                     - tf.tensordot(y_pred, g, [[1], [0]])))

    return tf.keras.models.load_model(
        path, custom_objects={"emd_loss": emd_loss, "kl_loss": kl_loss, "age_mae": age_mae})


def read_heads(prediction):
    """Keras hands back a dict; ONNX Runtime hands back a list in graph order."""
    if isinstance(prediction, dict):
        return np.asarray(prediction["age"])[0], np.asarray(prediction["gender"])[0]
    return np.asarray(prediction[0])[0], np.asarray(prediction[1])[0]


def predict(image_bgr, model, detector, size=IMG_SIZE):
    """-> (result dict, the crop the model saw). Raises NoFaceDetected.

    The crop is returned alongside the numbers so a caller can show what was
    actually looked at, rather than asking anyone to trust a bare figure.
    """
    face = align_crop(image_bgr, detector, size=size)
    if face is None:
        raise NoFaceDetected("no face detected in the image")
    batch = face[None].astype("float32") / 255.0
    dist, gender = read_heads(model.predict(batch, verbose=0))
    return {
        "age": round(float(dist @ age_grid(NUM_BINS)), 1),
        "gender": GENDERS[int(np.argmax(gender))],
        "gender_confidence": round(float(np.max(gender)), 4),
    }, face
