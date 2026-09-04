"""Shared pieces of the age/gender pipeline.

The scripts under `scripts/` were standalone and duplicated three things between
them: the Gaussian label construction, the stratified split, and the MediaPipe
crop. Each is pure enough to test on its own, so they live here and the scripts
import them. Nothing in this package imports TensorFlow, which is what lets CI
run the test suite without a 600MB install.
"""
