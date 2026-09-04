"""Age labels as distributions rather than scalars.

A 30-year-old face is barely distinguishable from a 31-year-old one and the
annotation itself is uncertain, so an exact age is a poor target. Each age is
spread as a Gaussian over the bin grid instead, and the predicted age is read
back as the expectation of that distribution.
"""
import numpy as np

NUM_BINS = 101
SIGMA = 2.0


def age_grid(num_bins=NUM_BINS):
    return np.arange(num_bins, dtype=np.float32)


def ages_to_distributions(ages, num_bins=NUM_BINS, sigma=SIGMA):
    """Exact ages -> (n, num_bins) row-normalised Gaussian distributions."""
    ages = np.asarray(ages, dtype=np.float32).reshape(-1)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    diffs = age_grid(num_bins)[None, :] - ages[:, None]
    dist = np.exp(-(diffs ** 2) / (2 * sigma ** 2))
    return (dist / dist.sum(1, keepdims=True)).astype("float32")


def expected_age(dist, num_bins=NUM_BINS):
    """Distributions -> the age each one points at, in years."""
    dist = np.asarray(dist, dtype=np.float32).reshape(-1, num_bins)
    return dist @ age_grid(num_bins)
