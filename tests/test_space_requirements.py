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
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = os.path.join(ROOT, "space", "README.md")
REQS = os.path.join(ROOT, "space", "requirements.txt")
SERVE = os.path.join(ROOT, "requirements-serve.txt")
MODEL = os.path.join(ROOT, "artifacts", "deep.keras")

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


def version_tuple(v):
    return tuple(int(p) for p in re.findall(r"\d+", v))


def test_gradio_is_pinned_to_the_version_the_space_card_declares():
    declared = card_field("sdk_version")
    assert declared, "space/README.md has no sdk_version"
    pinned = pin("gradio")
    assert pinned, "space/requirements.txt does not pin gradio"
    op, version = pinned
    assert op == "==", f"gradio must be pinned exactly, got '{op}{version}'"
    assert version == declared, f"card says gradio {declared}, requirements say {version}"


@SERVING
def test_keras_floor_is_at_least_what_the_saved_model_needs(reqs):
    if not os.path.exists(MODEL):
        pytest.skip("artifacts/deep.keras not present (run scripts/train.py)")
    with zipfile.ZipFile(MODEL) as z:
        saved = re.search(r'"keras_version"\s*:\s*"([^"]+)"',
                          z.read("metadata.json").decode("utf8")).group(1)
    pinned = pin("keras", reqs)
    assert pinned, f"{os.path.basename(reqs)} does not pin keras"
    op, floor = pinned
    assert ">" in op, f"keras needs a lower bound, got '{op}{floor}'"
    assert version_tuple(floor) >= version_tuple(saved), (
        f"model was saved by Keras {saved} but {os.path.basename(reqs)} "
        f"floors keras at {floor}")


@SERVING
def test_serving_images_do_not_install_training_only_packages(reqs):
    with open(reqs, encoding="utf8") as fh:
        body = fh.read()
    assert not re.search(r"^-r\s", body, re.M), (
        f"{os.path.basename(reqs)} uses `-r`, which drags the training "
        "requirements into a serving image -- list what it needs explicitly")
    for package in ("keras-tuner", "scikit-learn", "matplotlib", "pandas"):
        assert not re.search(rf"^{re.escape(package)}\b", body, re.M), \
            f"{package} is training-only and should not be in a serving image"
