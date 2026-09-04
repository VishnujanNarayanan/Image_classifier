"""Train/validation split that keeps both labels balanced.

Splitting at random lets a whole age band or one gender land mostly on one side,
which then shows up as a metric that moves for reasons nothing to do with the
model. Indices are bucketed by (age decade, gender) and split within each bucket.
"""
import numpy as np


def stratified_split(ages, genders, frac=0.8, seed=42):
    """-> (train_idx, val_idx), split `frac`/1-`frac` inside each bucket.

    Buckets holding fewer than three samples go entirely to train, so validation
    never contains a group the model has never seen.
    """
    ages = np.asarray(ages)
    genders = np.asarray(genders)
    if len(ages) != len(genders):
        raise ValueError("ages and genders must be the same length")
    rng = np.random.default_rng(seed)
    groups = {}
    for i, key in enumerate(zip(np.clip(ages // 10, 0, 10).astype(int), genders)):
        groups.setdefault(key, []).append(i)
    train, val = [], []
    for _, idx in sorted(groups.items()):
        idx = np.array(idx)
        rng.shuffle(idx)
        if len(idx) > 2:
            cut = int(frac * len(idx))
            train += list(idx[:cut])
            val += list(idx[cut:])
        else:
            train += list(idx)
    return np.array(sorted(train)), np.array(sorted(val))
