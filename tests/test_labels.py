import numpy as np
import pytest

from agc.labels import NUM_BINS, ages_to_distributions, expected_age


def test_rows_are_probability_distributions():
    d = ages_to_distributions([0, 30, 100])
    assert d.shape == (3, NUM_BINS)
    assert np.allclose(d.sum(axis=1), 1.0)
    assert (d >= 0).all()


@pytest.mark.parametrize("age", [1, 24, 47, 99])
def test_mass_peaks_on_the_true_age(age):
    assert int(ages_to_distributions([age])[0].argmax()) == age


def test_expectation_recovers_the_age_it_was_built_from():
    ages = np.array([5, 20, 45, 70], dtype="float32")
    # the grid is clipped at both ends, so ages near 0 or 100 pull inward
    assert np.allclose(expected_age(ages_to_distributions(ages)), ages, atol=0.05)


def test_a_wider_sigma_spreads_more_mass():
    peak_narrow = ages_to_distributions([40], sigma=1.0)[0].max()
    peak_wide = ages_to_distributions([40], sigma=6.0)[0].max()
    assert peak_wide < peak_narrow


def test_sigma_must_be_positive():
    with pytest.raises(ValueError):
        ages_to_distributions([30], sigma=0)


def test_ages_at_the_grid_edges_pull_inward():
    """The Gaussian is truncated at 0 and 100, so the tails have nowhere to go.

    Age 100 reads back as ~98.7 and no amount of training fixes that -- it is the
    representation, not the model. Worth pinning: it is the one place where the
    label round trip is lossy, and it explains a chunk of the 70+ band's error.
    """
    assert expected_age(ages_to_distributions([100]))[0] == pytest.approx(98.7, abs=0.1)
    assert expected_age(ages_to_distributions([0]))[0] == pytest.approx(1.3, abs=0.1)
    # everything away from the edges round-trips essentially exactly
    inner = np.arange(7, 94)
    assert np.allclose(expected_age(ages_to_distributions(inner)), inner, atol=0.01)
