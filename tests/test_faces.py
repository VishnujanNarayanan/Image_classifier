"""Crop geometry, tested against a stub detector.

MediaPipe is not installed on CI and its model is a black box anyway; what needs
checking is the margin arithmetic and the clamping around it, which is ours.
"""
import numpy as np
import pytest

pytest.importorskip("cv2", reason="agc.faces needs OpenCV")

from agc.faces import align_crop  # noqa: E402


class _Box:
    def __init__(self, xmin, ymin, width, height):
        self.xmin, self.ymin, self.width, self.height = xmin, ymin, width, height


class _Detection:
    def __init__(self, box):
        self.location_data = type("L", (), {"relative_bounding_box": box})()


class _Stub:
    """Returns a fixed relative box, or nothing at all."""

    def __init__(self, box=None):
        self.box = box

    def process(self, _rgb):
        return type("R", (), {"detections": [_Detection(self.box)] if self.box else []})()


IMG = np.full((200, 200, 3), 127, dtype=np.uint8)


def test_no_detection_returns_none():
    assert align_crop(IMG, _Stub()) is None


def test_crop_is_resized_to_the_requested_square():
    out = align_crop(IMG, _Stub(_Box(0.25, 0.25, 0.5, 0.5)), size=64)
    assert out.shape == (64, 64, 3)


def test_margin_widens_the_box():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[80:120, 80:120] = 255                      # a white square as the "face"
    tight = align_crop(img, _Stub(_Box(0.4, 0.4, 0.2, 0.2)), size=32, margin=0.0)
    loose = align_crop(img, _Stub(_Box(0.4, 0.4, 0.2, 0.2)), size=32, margin=0.25)
    # the tight crop is all face; the loose one has dragged in black background
    assert tight.mean() > loose.mean()


def test_a_box_running_off_the_edge_is_clamped_not_wrapped():
    out = align_crop(IMG, _Stub(_Box(-0.2, -0.2, 0.5, 0.5)), size=16, margin=0.1)
    assert out is not None and out.shape == (16, 16, 3)


def test_a_degenerate_box_returns_none():
    assert align_crop(IMG, _Stub(_Box(0.0, 0.0, 0.0, 0.0)), margin=0.0) is None
