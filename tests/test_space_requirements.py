"""Consistency checks on what the Hugging Face Space will install.

Both of these failed silently before: an unpinned gradio can land 5.x inside the
4.44 base image the Space card asks for, and an unpinned keras can be too old to
deserialize the saved model. Neither shows up until the Space is already built,
so they are cheaper to catch here.
"""
import os
import re
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = os.path.join(ROOT, "space", "README.md")
REQS = os.path.join(ROOT, "space", "requirements.txt")
MODEL = os.path.join(ROOT, "artifacts", "deep.keras")


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


def test_keras_floor_is_at_least_what_the_saved_model_needs():
    if not os.path.exists(MODEL):
        pytest.skip("artifacts/deep.keras not present (run scripts/train.py)")
    with zipfile.ZipFile(MODEL) as z:
        saved = re.search(r'"keras_version"\s*:\s*"([^"]+)"',
                          z.read("metadata.json").decode("utf8")).group(1)
    pinned = pin("keras")
    assert pinned, "space/requirements.txt does not pin keras"
    op, floor = pinned
    assert ">" in op, f"keras needs a lower bound, got '{op}{floor}'"
    assert version_tuple(floor) >= version_tuple(saved), (
        f"model was saved by Keras {saved} but the Space floors keras at {floor}")


def test_the_space_does_not_install_training_only_packages():
    with open(REQS, encoding="utf8") as fh:
        body = fh.read()
    for package in ("keras-tuner", "scikit-learn", "matplotlib"):
        assert not re.search(rf"^{re.escape(package)}\b", body, re.M), \
            f"{package} is training-only and should not be in the Space image"
