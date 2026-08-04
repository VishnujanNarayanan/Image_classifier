<h1 align="center">Age &amp; Gender Classifier</h1>

<p align="center">
  A multi-task CNN over UTKFace that predicts gender and age from one face crop —<br>
  age treated as a distribution over 101 bins and trained with an Earth Mover's Distance loss.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white"/>
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white"/>
  <img alt="Keras Tuner" src="https://img.shields.io/badge/Keras_Tuner-search-D00000?logo=keras&logoColor=white"/>
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-0.10.9-0097A7?logo=google&logoColor=white"/>
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-face_crop-5C3EE8?logo=opencv&logoColor=white"/>
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.21+-013243?logo=numpy&logoColor=white"/>
  <a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/badge/License-MIT-750014"/></a>
  <br>
  <a href="https://susanqq.github.io/UTKFace/"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-UTKFace-4C8CBF?style=for-the-badge"/></a>
  <br>
  <a href="https://github.com/VishnujanNarayanan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-VishnujanNarayanan-181717?logo=github&logoColor=white&style=for-the-badge"/></a>
  <a href="https://www.linkedin.com/in/vishnujan-narayanan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Vishnujan_Narayanan-0A66C2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg%3D%3D&logoColor=white&style=for-the-badge"/></a>
  <a href="https://substack.com/@vishnujannarayanan"><img alt="Substack" src="https://img.shields.io/badge/Substack-@vishnujannarayanan-FF6719?logo=substack&logoColor=white&style=for-the-badge"/></a>
</p>

<p align="center">
  🎯 <a href="#why-this-project-exists">Why</a> ·
  🧩 <a href="#architecture">Architecture</a> ·
  🧠 <a href="#design-decisions">Design Decisions</a> ·
  📊 <a href="#results">Results</a> ·
  ⚡ <a href="#installation">Installation</a> ·
  ⚠️ <a href="#limitations">Limitations</a> ·
  🗺️ <a href="#roadmap">Roadmap</a>
</p>

---

## Why this project exists

Age prediction is usually framed as regression on a single number, which throws away something
true about the label: a 30-year-old face is barely distinguishable from a 31-year-old one, and
the annotation itself is uncertain. Squared error treats a 1-year miss and a 20-year miss as
differing only in magnitude, not in kind.

This project instead represents each age as a **Gaussian distribution over 101 bins** and trains
with **Earth Mover's Distance**, a loss that knows bin 30 is close to bin 31 and far from bin 60.
Gender is learned jointly from the same convolutional trunk, so both tasks share one face
representation.

## Features

- **Multi-task CNN** — one trunk, two heads: 101-way age distribution and 2-way gender.
- **Label Distribution Learning** — exact ages converted to Gaussians with σ = 2.0.
- **EMD loss** over cumulative distributions, so error scales with distance between age bins.
- **MediaPipe face detection** with a 5% margin crop, replacing naive centre-cropping.
- **Stratified splitting** by age decade × gender, so both are balanced across train and val.
- **Keras Tuner search** over filters, kernel size, dense width, dropout, and learning rate.
- **Interpretable age metric** — MAE in years, recovered as the expectation of the predicted
  distribution rather than an argmax.

## Architecture

```mermaid
flowchart TB
    F["UTKFace JPEGs<br/>age_gender_*.jpg"] --> P["Parse filename<br/>age, gender"]
    F --> D["MediaPipe FaceDetection<br/>confidence 0.6"]
    D -->|no face| Skip["Dropped"]
    D --> C["Crop + 5% margin<br/>resize 64x64, scale to 0-1"]
    P --> LDL["ages_to_distributions<br/>Gaussian sigma=2 over 0..100"]

    C --> S["Stratified split<br/>age decade x gender, 80/20"]
    LDL --> S

    S --> M["CNN trunk<br/>Conv2D -> BatchNorm -> MaxPool -> Flatten<br/>Dense -> Dropout"]
    M --> A["age head<br/>Dense(101, softmax)"]
    M --> G["gender head<br/>Dense(2, softmax)"]

    A --> LA["emd_loss<br/>weight ALPHA=1.0"]
    G --> LG["categorical_crossentropy<br/>weight BETA=1.0"]
    A --> MAE["mae_from_distribution<br/>expectation over bins, in years"]
```

## Design Decisions

**Age is a distribution, not a scalar.** `ages_to_distributions` places a Gaussian with σ = 2.0
over the 0–100 grid and normalises it. A label of 30 becomes soft mass across roughly 26–34,
which matches how uncertain the annotation actually is.

**EMD compares cumulative distributions.** `emd_loss` takes `cumsum` of both the true and
predicted distributions and returns the mean absolute difference. Predicting 32 for a 30-year-old
is penalised far less than predicting 60 — cross-entropy over bins would treat both as simply
"wrong".

**Predicted age is the expectation, not the argmax.** `mae_from_distribution` dots the predicted
distribution with the age grid. This uses the whole distribution, and gives a metric in years that
means something to a reader.

**Faces are detected, not assumed centred.** MediaPipe finds the face and the crop is expanded 5%
on each side to keep the jaw and hairline, both of which carry age signal. Images with no detected
face are dropped rather than fed in as noise.

**The split is stratified on both labels jointly.** Indices are bucketed by
`(age // 10, gender)` and split 80/20 within each bucket, so no age band or gender is
concentrated on one side. Buckets with fewer than three samples go entirely to train, so
validation never contains a group the model has never seen.

**Both tasks are weighted equally** (`ALPHA = BETA = 1.0`), which is a starting point rather than
a tuned choice — see [Limitations](#limitations).

## Results

Final epoch of a 15-epoch run at 64×64 input, batch size 64:

| Metric | Train | Validation |
|---|---|---|
| Gender accuracy | 0.9938 | **0.8167** |
| Age MAE (years) | 11.59 | **13.22** |
| True-age MAE via `ValReporter` | — | **12.80 years** |
| Age (EMD) loss | 0.1189 | 0.1348 |
| Gender loss | 0.0301 | 0.5454 |
| Total loss | 0.1492 | 0.6685 |

**These numbers show clear overfitting on the gender head.** Training accuracy reaches 99.4%
while validation stalls at 81.7%, and validation gender loss *rises* from 0.50 to 0.55 over the
final epochs while training loss falls to 0.03. The age head generalises better — an 11.6/13.2
train-to-val gap is comparatively modest — but a 12.8-year MAE is weak in absolute terms.

For reference, the tuner's first search trial started at ~28.5 years validation MAE and 74.1%
gender accuracy, so training does move both substantially; the ceiling here is architectural, not
a failure to converge.

## Project Structure

```
Age_Gender_classifier/
├── Image_Classifier!.ipynb           # Main pipeline: MediaPipe crop, LDL, EMD, tuner, training
├── Image_classifier_multitask.ipynb  # Multi-task variant
├── image_classifier2_multitask.ipynb # Multi-task variant
├── image3.ipynb                      # Earlier experiment
├── Image_classifier.ipynb            # Earlier experiment
├── image_classifier2.ipynb           # Earlier experiment
├── extract.py                        # Flattens UTKFace .tar.gz parts into one image directory
├── UTKFace/images_flat/part1/        # Expected image location (not committed)
├── tuner/age_gender/                 # Keras Tuner search state
├── requirements.txt
├── LICENSE.txt
└── README.md
```

`Image_Classifier!.ipynb` is the current pipeline; the other notebooks are earlier iterations kept
for reference.

## Installation

Clone the repository:

```bash
git clone https://github.com/VishnujanNarayanan/Image_classifier.git
cd Image_classifier
```

Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate      # Linux / macOS
env\Scripts\activate         # Windows
pip install -r requirements.txt
pip install mediapipe==0.10.9 keras-tuner
```

`mediapipe` and `keras-tuner` are installed by the notebook's first cells but are **not** listed
in `requirements.txt`.

### Getting the data

Download [UTKFace](https://susanqq.github.io/UTKFace/). Filenames encode the labels as
`age_gender_race_date.jpg`, which is what the loader parses. If the download arrives as `.tar.gz`
parts, `extract.py` flattens them into a single directory:

```bash
python extract.py    # edit source_dir and target_dir first — both are absolute Windows paths
```

Then point `DATA_DIR` in the notebook at the resulting folder.

## Usage

```bash
jupyter notebook "Image_Classifier!.ipynb"
```

Set `DATA_DIR` in the config cell, then run top to bottom. The notebook preprocesses every image
through MediaPipe, builds the age distributions, runs the tuner search, and trains the selected
model.

## Configuration

All settings live in one config cell:

| Setting | Value | Meaning |
|---|---|---|
| `DATA_DIR` | absolute path | Directory of UTKFace JPEGs |
| `IMG_SIZE` | 64 | Input resolution after cropping |
| `SIGMA` | 2.0 | Gaussian spread for label distributions |
| `ALPHA` | 1.0 | Age loss weight |
| `BETA` | 1.0 | Gender loss weight |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 15 | Training epochs |
| `AGE_MAX` | 100 | Ages clipped to this maximum |
| `NUM_BINS` | 101 | Age bins, 0–100 inclusive |
| Face detection confidence | 0.6 | `min_detection_confidence` |
| Crop margin | 5% | Expansion around the detected box |

### Tuner search space

| Hyperparameter | Values |
|---|---|
| `filters1` | 32, 64, 128 |
| `kernel1` | 3, 5 |
| `dense_units` | 128, 256, 512 |
| `dropout` | 0.2 – 0.5, step 0.1 |
| `lr` | 1e-2, 1e-3, 1e-4, 1e-5 |

## Example Workflow

1. Download UTKFace and flatten it into one directory of JPEGs.
2. Point `DATA_DIR` at that directory.
3. Run the loading cell. Each filename is split on `_` to recover age and gender; ages are
   clipped to 0–100; MediaPipe crops the face. Images with no detected face are dropped and
   counted in `bad_files`.
4. Run `ages_to_distributions` to convert exact ages into 101-bin Gaussians.
5. Run the split cell — indices are grouped by `(age decade, gender)` and split 80/20 within each
   group.
6. Run the tuner, then train the selected model. `ValReporter` prints true-age MAE in years after
   every epoch.
7. Run the plotting cell for MAE, accuracy, and loss curves.

## Dependencies

| Package | Why |
|---|---|
| `tensorflow` / `keras` | Model definition, custom EMD loss, training loop |
| `keras-tuner` | Hyperparameter search over the CNN and optimiser |
| `mediapipe` | Face detection for cropping |
| `opencv-python` | Image reading, colour conversion, resizing |
| `numpy` | Label distribution construction and array handling |
| `scikit-learn` | `train_test_split`, MAE and accuracy helpers |
| `matplotlib` | Training curves |

## Limitations

- **The gender head overfits badly.** 99.4% train against 81.7% validation accuracy, with
  validation loss rising while training loss falls.
- **12.8-year age MAE is weak.** Published UTKFace results using deeper backbones reach
  substantially lower error; this architecture is one convolutional block.
- **The trunk is very shallow** — a single `Conv2D → BatchNorm → MaxPool` before flattening. The
  tuner searches its width, not its depth.
- **64×64 input discards fine detail** — skin texture and wrinkles are exactly the age signal.
- **No data augmentation.** No flips, rotations, brightness jitter, or scale jitter, which is
  likely the single largest contributor to the overfitting.
- **`Flatten` after one pooling layer** produces a very wide dense layer, concentrating almost all
  parameters in one place.
- **Loss weights are untuned.** `ALPHA = BETA = 1.0` was never varied, despite the two losses
  operating on different scales — EMD sits around 0.12 while gender cross-entropy reaches 0.03.
- **`DATA_DIR` is an absolute Windows path** and must be edited before the notebook runs.
- **`extract.py` has hardcoded absolute paths** and does not accept arguments.
- **`mediapipe` and `keras-tuner` are missing from `requirements.txt`**, and no versions are
  pinned.
- **`UTKFace/images_flat/part1/` is empty in this checkout** — the data is not committed.
- **The tuner-selected model and the manually built `model` are trained in separate cells**
  (cells 13 and 14), so which model produced the reported figures depends on execution order.
- **No model is saved.** Weights exist only in the kernel.
- **Six notebooks are committed**, most of them superseded, which makes the entry point unclear.
- **Ethnicity is present in the UTKFace filenames but unused.** Any deployment should consider
  whether accuracy is uniform across groups — that is not measured here.

## Roadmap

- Add augmentation: horizontal flips, small rotations, brightness and scale jitter.
- Deepen the trunk to three or four convolutional blocks with global average pooling instead of
  `Flatten`.
- Raise input resolution to 128×128.
- Add early stopping and a learning-rate schedule on validation loss.
- Tune `ALPHA` / `BETA`, or normalise the two losses onto a common scale.
- Try transfer learning from a pretrained face backbone.
- Move `DATA_DIR` and the `extract.py` paths into arguments.
- Pin all dependencies, including `mediapipe` and `keras-tuner`.
- Save the trained model and add a standalone inference script.
- Report accuracy broken down by the ethnicity label to check for uneven performance.
- Consolidate the six notebooks down to one.

## License

Released under the MIT License — see [LICENSE.txt](LICENSE.txt).

The UTKFace dataset is provided by its authors for **non-commercial research purposes only**;
its terms apply independently of this repository's licence.

## Acknowledgements

[UTKFace](https://susanqq.github.io/UTKFace/) — over 20,000 face images annotated with age,
gender, and ethnicity (Zhang, Song &amp; Qi).

## Author

<p align="center">
  <strong>Vishnujan Narayanan</strong>
</p>

<p align="center">
  <a href="https://github.com/VishnujanNarayanan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-VishnujanNarayanan-181717?logo=github&logoColor=white&style=for-the-badge"/></a>
  <a href="https://www.linkedin.com/in/vishnujan-narayanan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Vishnujan_Narayanan-0A66C2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg%3D%3D&logoColor=white&style=for-the-badge"/></a>
  <a href="https://substack.com/@vishnujannarayanan"><img alt="Substack" src="https://img.shields.io/badge/Substack-@vishnujannarayanan-FF6719?logo=substack&logoColor=white&style=for-the-badge"/></a>
</p>
