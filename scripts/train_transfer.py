"""Transfer-learning variant: an ImageNet backbone in place of the scratch trunk.

Same task formulation as train.py - age as a Gaussian label distribution over 101
one-year bins trained with an EMD loss, gender as a softmax, both heads sharing a
trunk. The difference is the trunk: MobileNetV2 pretrained on ImageNet at 128x128
rather than a few conv blocks learned from 8k faces. Distinguishing a 55-year-old
from a 75-year-old is a fine-texture problem, and that is what a scratch CNN of
this size on this much data could not represent - both the ceiling at 35 and the
32-year error past 60 came from that, not from the loss.

Two stages: head-only with the backbone frozen, then fine-tuning the top of the
backbone at a low learning rate.
"""
import os, json, numpy as np, tensorflow as tf

HERE     = os.path.dirname(__file__)
CACHE    = os.path.join(HERE, "..", "cache_faces_128.npz")
OUT_DIR  = os.path.join(HERE, "..", "artifacts")
IMG      = 128
NUM_BINS = 101
AGE_GRID = np.arange(NUM_BINS, dtype=np.float32)
SIGMA    = 2.0
ALPHA    = 5.0
BATCH    = 64
SEED     = 42
E_HEAD   = int(os.environ.get("E_HEAD", 12))
E_FINE   = int(os.environ.get("E_FINE", 30))

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED)
print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)

d       = np.load(CACHE, allow_pickle=True)
Xu      = d["X"]                       # uint8, cast per batch to keep host RAM down
ages    = d["ages"].astype("float32")
genders = d["genders"].astype("int32")
paths   = d["paths"]
print("data:", Xu.shape, flush=True)

diffs    = AGE_GRID[None, :] - ages[:, None]
age_dist = np.exp(-(diffs ** 2) / (2 * SIGMA ** 2))
age_dist /= age_dist.sum(1, keepdims=True)
age_dist = age_dist.astype("float32")
gen_oh   = np.eye(2, dtype="float32")[genders]

rng = np.random.default_rng(SEED)
tr, va = [], []
groups = {}
for i, k in enumerate(zip(np.clip(ages // 10, 0, 10).astype(int), genders)):
    groups.setdefault(k, []).append(i)
for k, idx in groups.items():
    idx = np.array(idx); rng.shuffle(idx)
    if len(idx) > 2:
        c = int(0.8 * len(idx)); tr += list(idx[:c]); va += list(idx[c:])
    else:
        tr += list(idx)
tr, va = np.array(tr), np.array(va)
print(f"train {len(tr)}  val {len(va)}", flush=True)


def emd_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return tf.reduce_mean(tf.abs(tf.cumsum(y_true, 1) - tf.cumsum(y_pred, 1)))

def age_mae(y_true, y_pred):
    g = tf.constant(AGE_GRID)
    return tf.reduce_mean(tf.abs(tf.tensordot(y_true, g, [[1], [0]])
                                 - tf.tensordot(y_pred, g, [[1], [0]])))

aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                           tf.keras.layers.RandomRotation(0.06),
                           tf.keras.layers.RandomZoom(0.12),
                           tf.keras.layers.RandomContrast(0.15)])

def ds(idx, training):
    t = tf.data.Dataset.from_tensor_slices(
        (Xu[idx], {"age": age_dist[idx], "gender": gen_oh[idx]}))
    t = t.map(lambda a, b: (tf.cast(a, tf.float32), b), tf.data.AUTOTUNE)
    if training:
        t = t.shuffle(2048, seed=SEED).map(
            lambda a, b: (aug(a, training=True), b), tf.data.AUTOTUNE)
    # MobileNetV2 expects inputs scaled to [-1, 1], not [0, 1]
    t = t.map(lambda a, b: (tf.keras.applications.mobilenet_v2.preprocess_input(a), b),
              tf.data.AUTOTUNE)
    return t.batch(BATCH).prefetch(tf.data.AUTOTUNE)


base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
base.trainable = False

inp = tf.keras.Input((IMG, IMG, 3))
x   = base(inp, training=False)
x   = tf.keras.layers.GlobalAveragePooling2D()(x)
x   = tf.keras.layers.Dropout(0.3)(x)
x   = tf.keras.layers.Dense(256, activation="relu")(x)
x   = tf.keras.layers.Dropout(0.3)(x)
age = tf.keras.layers.Dense(NUM_BINS, activation="softmax", name="age")(x)
gen = tf.keras.layers.Dense(2, activation="softmax", name="gender")(x)
m   = tf.keras.Model(inp, {"age": age, "gender": gen})

def compile_at(lr):
    m.compile(tf.keras.optimizers.Adam(lr),
              loss={"age": emd_loss, "gender": "categorical_crossentropy"},
              loss_weights={"age": ALPHA, "gender": 1.0},
              metrics={"age": [age_mae], "gender": ["accuracy"]})

train_ds, val_ds = ds(tr, True), ds(va, False)

print("\n=== stage 1: frozen backbone ===", flush=True)
compile_at(1e-3)
m.fit(train_ds, validation_data=val_ds, epochs=E_HEAD, verbose=2)

print("\n=== stage 2: fine-tune top of backbone ===", flush=True)
base.trainable = True
for l in base.layers[:-40]:
    l.trainable = False
compile_at(1e-4)
m.fit(train_ds, validation_data=val_ds, epochs=E_FINE, verbose=2,
      callbacks=[tf.keras.callbacks.EarlyStopping(
                     "val_loss", patience=8, restore_best_weights=True, mode="min"),
                 tf.keras.callbacks.ReduceLROnPlateau(
                     "val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)])

pr = m.predict(val_ds, verbose=0)
pa = (pr["age"] * AGE_GRID).sum(1)
pg = pr["gender"].argmax(1)
mae = float(np.mean(np.abs(pa - ages[va])))
acc = float(np.mean(pg == genders[va]))
print(f"\ntransfer: val age MAE {mae:.2f} years | gender acc {acc:.4f} "
      f"| params {m.count_params():,}", flush=True)
print(f"pred range {pa.min():.1f}-{pa.max():.1f} (true {ages[va].min():.0f}-{ages[va].max():.0f})")
for lo, hi in [(0,12),(13,19),(20,29),(30,44),(45,59),(60,120)]:
    k = (ages[va] >= lo) & (ages[va] <= hi)
    print("  %2d-%3d n=%4d  MAE %5.1f  gender %.3f"
          % (lo, hi, k.sum(), np.mean(np.abs(pa[k]-ages[va][k])), np.mean(pg[k]==genders[va][k])))

m.save(os.path.join(OUT_DIR, "transfer.keras"))
np.savez_compressed(os.path.join(OUT_DIR, "val_preds_transfer.npz"),
                    idx=va, pred_age=pa, pred_gender=pg,
                    true_age=ages[va], true_gender=genders[va], paths=paths[va])
json.dump({"model": "mobilenetv2-128", "mae": mae, "acc": acc,
           "n_train": int(len(tr)), "n_val": int(len(va))},
          open(os.path.join(OUT_DIR, "metrics_transfer.json"), "w"), indent=2)
print("wrote", OUT_DIR, flush=True)
