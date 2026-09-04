"""Convert the trained Keras model to ONNX, and prove the conversion is faithful.

Serving needed ~1GB resident just to import TensorFlow, which does not fit the
512MB a no-card free tier gives you. ONNX Runtime loads the same graph in a
fraction of that, so the serving path drops TensorFlow entirely.

A conversion that "succeeded" is not the same as a conversion that is correct, so
this does not stop at writing the file: it runs both models over real cached
faces and fails loudly if their outputs diverge.

    python scripts/export_onnx.py                 # artifacts/deep.keras -> deep.onnx
    python scripts/export_onnx.py --tolerance 1e-4
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_IN = os.path.join(ROOT, "artifacts", "deep.keras")
DEFAULT_OUT = os.path.join(ROOT, "artifacts", "deep.onnx")
CACHE = os.path.join(ROOT, "cache_faces.npz")


def export(keras_path, onnx_path):
    """Write `keras_path` out as ONNX. Returns the loaded Keras model."""
    from agc.inference import load_model
    model = load_model(keras_path)

    # Keras traces the graph from a real call, and a model restored from disk has
    # not been called yet -- export fails with "the model provided has never
    # called" unless it is warmed up first.
    model.predict(np.zeros((1, 64, 64, 3), dtype="float32"), verbose=0)

    # Keras 3 can emit ONNX directly. Fall back to tf2onnx.convert.from_keras
    # (note: this version has no from_saved_model) and say which route was taken,
    # because the two do not produce identically-named outputs.
    try:
        model.export(onnx_path, format="onnx")
        print("exported via keras model.export(format='onnx')")
    except Exception as exc:
        print(f"keras onnx export unavailable ({type(exc).__name__}: {exc}), using tf2onnx")
        import tf2onnx
        tf2onnx.convert.from_keras(model, output_path=onnx_path)
        print("exported via tf2onnx.convert.from_keras")
    return model


def sample_faces(n):
    """Real cached crops if they exist, otherwise noise of the right shape."""
    if os.path.exists(CACHE):
        X = np.load(CACHE, allow_pickle=True)["X"]
        idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
        return X[idx].astype("float32") / 255.0
    print("cache_faces.npz absent -- verifying on random input instead")
    return np.random.default_rng(0).random((n, 64, 64, 3)).astype("float32")


def verify(keras_model, onnx_path, faces, tolerance):
    """Fail loudly if the two models disagree on real faces."""
    import onnxruntime as ort
    from agc.inference import read_heads

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    print(f"onnx input  : {name} {sess.get_inputs()[0].shape}")
    print(f"onnx outputs: {[o.name for o in sess.get_outputs()]}")

    worst_age = worst_gen = 0.0
    for face in faces:
        batch = face[None]
        k_age, k_gen = read_heads(keras_model.predict(batch, verbose=0))
        outs = sess.run(None, {name: batch})
        o_age, o_gen = read_heads(outs)
        worst_age = max(worst_age, float(np.abs(k_age - o_age).max()))
        worst_gen = max(worst_gen, float(np.abs(k_gen - o_gen).max()))

    print(f"max abs difference over {len(faces)} faces -- "
          f"age {worst_age:.3e}  gender {worst_gen:.3e}")
    if max(worst_age, worst_gen) > tolerance:
        sys.exit(f"FAILED: divergence exceeds tolerance {tolerance:g}")
    print("conversion verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", default=DEFAULT_IN)
    ap.add_argument("--onnx", default=DEFAULT_OUT)
    ap.add_argument("--faces", type=int, default=64)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    model = export(args.keras, args.onnx)
    size = os.path.getsize(args.onnx) / 1e6
    print(f"wrote {args.onnx} ({size:.1f} MB)")
    verify(model, args.onnx, sample_faces(args.faces), args.tolerance)


if __name__ == "__main__":
    main()
