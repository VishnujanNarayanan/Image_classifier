import numpy as np
import pytest

from agc.split import stratified_split

rng = np.random.default_rng(0)
AGES = rng.integers(1, 90, size=600)
GENDERS = rng.integers(0, 2, size=600)


def test_split_is_a_partition():
    tr, va = stratified_split(AGES, GENDERS)
    assert len(set(tr) & set(va)) == 0
    assert sorted(list(tr) + list(va)) == list(range(len(AGES)))


def test_every_validation_decade_also_appears_in_train():
    tr, va = stratified_split(AGES, GENDERS)
    seen = set(zip(AGES[tr] // 10, GENDERS[tr]))
    assert set(zip(AGES[va] // 10, GENDERS[va])) <= seen


def test_split_is_deterministic_for_a_seed():
    a, _ = stratified_split(AGES, GENDERS, seed=7)
    b, _ = stratified_split(AGES, GENDERS, seed=7)
    c, _ = stratified_split(AGES, GENDERS, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_tiny_buckets_go_entirely_to_train():
    # one 90s male, alone in his bucket: he must not be the only sample of that
    # group and also be the thing the model is graded on
    ages = np.array([25] * 40 + [95])
    genders = np.array([0] * 40 + [0])
    tr, va = stratified_split(ages, genders)
    assert 40 in tr and 40 not in va


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        stratified_split([1, 2, 3], [0, 1])
