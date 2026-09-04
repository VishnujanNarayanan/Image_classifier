"""Detect, crop and cache faces from UTKFace.

Mirrors the notebook's align_crop: MediaPipe short-range detector, the box
expanded by 5% on every side, resized to 64x64. Caches the model input, the
labels and the source path so the grid renderer can show the original photo
rather than the 64px crop the model actually sees.
"""
import os, numpy as np, cv2, mediapipe as mp

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "UTKFace", "images_flat", "part1")
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        os.environ.get("CACHE", "cache_faces.npz"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 64))
AGE_MAX  = 100

det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def align_crop(img):
    h, w = img.shape[:2]
    res = det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    b = res.detections[0].location_data.relative_bounding_box
    x0, y0 = max(0, int((b.xmin - 0.05) * w)), max(0, int((b.ymin - 0.05) * h))
    x1, y1 = min(w, int((b.xmin + b.width + 0.05) * w)), min(h, int((b.ymin + b.height + 0.05) * h))
    crop = img[y0:y1, x0:x1]
    return None if crop.size == 0 else cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg"))
X, ages, genders, paths = [], [], [], []
skipped_label = skipped_read = skipped_noface = 0

for i, fn in enumerate(files):
    if i % 500 == 0:
        print(f"{i}/{len(files)}", flush=True)
    p = fn.split("_")
    try:
        age, gender = int(p[0]), int(p[1])
    except Exception:
        skipped_label += 1; continue
    if gender not in (0, 1):          # UTKFace ships one file labelled gender=3
        skipped_label += 1; continue
    img = cv2.imread(os.path.join(DATA_DIR, fn))
    if img is None:
        skipped_read += 1; continue
    face = align_crop(img)
    if face is None:
        skipped_noface += 1; continue
    X.append(face)
    ages.append(min(age, AGE_MAX))
    genders.append(gender)
    paths.append(fn)

X = np.asarray(X, dtype=np.uint8)
np.savez_compressed(OUT, X=X, ages=np.asarray(ages, np.int32),
                    genders=np.asarray(genders, np.int32), paths=np.asarray(paths))
print(f"\nkept {len(X)} of {len(files)}")
print(f"  bad label   {skipped_label}")
print(f"  unreadable  {skipped_read}")
print(f"  no face     {skipped_noface}")
print("->", OUT)
