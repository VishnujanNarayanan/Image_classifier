"""Gradio demo for the trained age/gender model.

The notebooks train and print; there is nothing to interact with. This wraps the
saved model in the same shape the classifier is actually used in - drop in a photo,
get a prediction - which is also what makes the project demonstrable on video.

    ./venv_tf/bin/python scripts/demo.py        # http://127.0.0.1:7861
"""
import os, numpy as np, cv2, mediapipe as mp, tensorflow as tf, gradio as gr

HERE     = os.path.dirname(__file__)
MODEL    = os.path.join(HERE, "..", "artifacts", os.environ.get("MODEL", "deep.keras"))
SAMPLES  = os.path.join(HERE, "..", "UTKFace", "images_flat", "part1")
IMG_SIZE = 64
NUM_BINS = 101
AGE_GRID = np.arange(NUM_BINS, dtype=np.float32)

det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def emd_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return tf.reduce_mean(tf.abs(tf.cumsum(y_true, 1) - tf.cumsum(y_pred, 1)))

def kl_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

def age_mae(y_true, y_pred):
    g = tf.constant(AGE_GRID)
    return tf.reduce_mean(tf.abs(tf.tensordot(y_true, g, [[1], [0]])
                                 - tf.tensordot(y_pred, g, [[1], [0]])))

model = tf.keras.models.load_model(MODEL, custom_objects={"emd_loss": emd_loss, "kl_loss": kl_loss, "age_mae": age_mae})
print("loaded", MODEL, flush=True)


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


def predict(rgb):
    if rgb is None:
        return "no image", "", None
    face = align_crop(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if face is None:
        return "no face detected", "", None
    pr = model.predict(face[None].astype("float32") / 255.0, verbose=0)
    dist = pr["age"][0] if isinstance(pr, dict) else pr[0][0]
    gend = pr["gender"][0] if isinstance(pr, dict) else pr[1][0]
    age = float((dist * AGE_GRID).sum())
    g = "Male" if int(np.argmax(gend)) == 0 else "Female"
    conf = float(np.max(gend))
    # the detected crop is returned so the demo shows what the model actually saw
    return f"{age:.0f} years", f"{g}  ({conf:.0%} confidence)", cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


examples = []
if os.path.isdir(SAMPLES):
    files = sorted(os.listdir(SAMPLES))
    for want in (3, 16, 26, 35, 51, 68):        # spread across the age range
        hit = next((f for f in files if f.split("_")[0] == str(want)), None)
        if hit:
            examples.append(os.path.join(SAMPLES, hit))

with gr.Blocks(title="Age & Gender Classifier") as demo:
    gr.Markdown("# Age & Gender Classifier\n"
                "Multi-task CNN over UTKFace. A face is detected and aligned, then one "
                "shared trunk predicts an age distribution and a gender.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="face photo", height=320)
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            out_age  = gr.Textbox(label="Predicted age")
            out_gen  = gr.Textbox(label="Predicted gender")
            out_face = gr.Image(label="Detected face (model input)", height=160)
    if examples:
        gr.Examples(examples=examples, inputs=inp, label="Sample faces")
    btn.click(predict, inp, [out_age, out_gen, out_face])
    inp.change(predict, inp, [out_age, out_gen, out_face])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7861)))
