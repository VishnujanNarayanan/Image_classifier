"""Consistency checks on what the SERVING images install.

There are two of them -- the Hugging Face Space and the Docker image -- and they
had the same two holes. An unpinned gradio can land 5.x inside the 4.44 base
image the Space card asks for, and an unpinned keras can be too old to
deserialize the saved model. Neither shows up until the image is already built,
and the keras one presents as a container that starts cleanly and then dies on
the first request.

requirements-serve.txt also carried a comment claiming training-only packages
were absent while `-r requirements.txt` was pulling in matplotlib, scikit-learn
and pandas. The last test here is what stops that drifting back.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = os.path.join(ROOT, "space", "README.md")
REQS = os.path.join(ROOT, "space", "requirements.txt")
SERVE = os.path.join(ROOT, "requirements-serve.txt")

#: Every requirements file that ends up in front of a user.
SERVING = pytest.mark.parametrize("reqs", [REQS, SERVE],
                                  ids=["space", "docker"])


def pin(package, path=REQS):
    """-> the version this requirements file pins `package` to, or None."""
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            m = re.match(rf"^{re.escape(package)}\s*([=><~!]+)\s*(.+)$", line)
            if m:
                return m.group(1), m.group(2).strip()
    return None


def card_field(name):
    with open(CARD, encoding="utf8") as fh:
        m = re.search(rf"^{name}:\s*(\S+)\s*$", fh.read(), re.M)
    return m.group(1) if m else None


def test_gradio_is_pinned_to_the_version_the_space_card_declares():
    declared = card_field("sdk_version")
    assert declared, "space/README.md has no sdk_version"
    pinned = pin("gradio")
    assert pinned, "space/requirements.txt does not pin gradio"
    op, version = pinned
    assert op == "==", f"gradio must be pinned exactly, got '{op}{version}'"
    assert version == declared, f"card says gradio {declared}, requirements say {version}"


def test_the_deployed_image_does_not_install_gradio():
    """Cold start is the constraint: gradio costs 1.6s of import and ~150MB RSS.

    space/requirements.txt is exempt -- a Hugging Face Space IS a Gradio app.
    This is about what Render and Docker run, which is the static page instead.
    """
    with open(SERVE, encoding="utf8") as fh:
        assert not re.search(r"^gradio\b", fh.read(), re.M), \
            "requirements-serve.txt installs gradio; the deployed UI is static"


@SERVING
def test_serving_pins_the_onnx_runtime_it_actually_uses(reqs):
    """agc/inference.py imports onnxruntime for any .onnx model, so it must ship."""
    with open(reqs, encoding="utf8") as fh:
        assert re.search(r"^onnxruntime\b", fh.read(), re.M), \
            f"{os.path.basename(reqs)} does not install onnxruntime"


@SERVING
def test_serving_images_do_not_install_training_only_packages(reqs):
    with open(reqs, encoding="utf8") as fh:
        body = fh.read()
    assert not re.search(r"^-r\s", body, re.M), (
        f"{os.path.basename(reqs)} uses `-r`, which drags the training "
        "requirements into a serving image -- list what it needs explicitly")
    # tensorflow and keras are on this list for a hard reason, not tidiness:
    # importing TensorFlow costs ~1GB resident and the free tier gives 512MB.
    for package in ("tensorflow", "tensorflow-cpu", "keras", "keras-tuner",
                    "scikit-learn", "matplotlib", "pandas"):
        assert not re.search(rf"^{re.escape(package)}\b", body, re.M), \
            f"{package} is training-only and must not be in a serving image"
