"""Face detection and cropping.

The detector is passed in rather than constructed here: MediaPipe is a heavy
import and a hard dependency to install on CI, and injecting it means the crop
geometry can be tested against a stub that returns a known box.
"""
import cv2

IMG_SIZE = 64
MARGIN = 0.05


def align_crop(img, detector, size=IMG_SIZE, margin=MARGIN):
    """-> a `size`x`size` BGR crop of the first detected face, or None.

    The box is expanded by `margin` on every side to keep the jaw and hairline,
    both of which carry age signal that a tight box throws away.
    """
    h, w = img.shape[:2]
    res = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    b = res.detections[0].location_data.relative_bounding_box
    x0 = max(0, int((b.xmin - margin) * w))
    y0 = max(0, int((b.ymin - margin) * h))
    x1 = min(w, int((b.xmin + b.width + margin) * w))
    y1 = min(h, int((b.ymin + b.height + margin) * h))
    crop = img[y0:y1, x0:x1]
    return None if crop.size == 0 else cv2.resize(crop, (size, size))


def detector(min_confidence=0.6):
    """The MediaPipe short-range detector the pipeline actually runs with."""
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_confidence)
