---
title: Age & Gender Classifier
emoji: 🧑
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Age & Gender Classifier

A multi-task CNN over [UTKFace](https://susanqq.github.io/UTKFace/) that predicts age
and gender from one face photo. Age is modelled as a distribution over 101 one-year
bins rather than a single number, and the prediction is the expectation of that
distribution.

Validation age MAE is **8.27 years** and gender accuracy **0.842** on 2,019 held-out
faces — honest numbers, not strong ones. Error runs from 4 years on the youngest band
to 12 on the worst. See the [GitHub repository](https://github.com/VishnujanNarayanan/Image_classifier)
for the per-band breakdown and the limitations.

Ethnicity is present in the UTKFace labels but unused here, and accuracy across groups
has **not** been measured. Do not use this for anything that matters.
