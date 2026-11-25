import math

import numpy as np
import pytest

from photoquest_eval.calibration import evaluate_class_oof, threshold_for_far
from photoquest_eval.config import FAR_TARGET, PROTOCOLS


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize(
    "negatives",
    [
        np.zeros(100, dtype=np.float32),
        np.array([0] * 94 + [0.1] * 6, dtype=np.float32),
        np.linspace(-1, 1, 100, dtype=np.float32),
    ],
)
def test_far_never_exceeds_target_on_calibration(negatives, protocol):
    cutoff = threshold_for_far(negatives, protocol)
    assert np.mean(negatives > cutoff) <= FAR_TARGET


def test_zero_scores_are_not_confirmed():
    cutoff = threshold_for_far(np.zeros(100), "empirical_5pct")
    assert cutoff == 0.0
    assert not np.any(np.zeros(10) > cutoff)


def test_conservative_is_not_looser_than_empirical():
    negatives = np.random.default_rng(0).random(1000)
    empirical = threshold_for_far(negatives, "empirical_5pct")
    conservative = threshold_for_far(negatives, "confidence_95pct")
    assert conservative >= empirical


def test_conservative_rejects_everything_on_tiny_sample():
    assert threshold_for_far(np.arange(10.0), "confidence_95pct") == math.inf


def test_unknown_protocol():
    with pytest.raises(ValueError):
        threshold_for_far(np.zeros(10), "other")


def test_oof_separable_classes():
    y = np.array([1] * 50 + [0] * 450)
    scores = np.where(y == 1, 0.9, 0.1) + np.random.default_rng(1).normal(0, 0.01, len(y))
    predicted, folds, calibration = evaluate_class_oof(y, scores, "empirical_5pct")
    assert set(folds) == {0, 1, 2, 3, 4}
    assert len(calibration) == 5
    assert predicted[y == 1].all()
    # Гарантия FAR действует на калибровочных фолдах; на отложенном он может слегка превысить цель.
    assert all(row["calibration_far"] <= FAR_TARGET for row in calibration)
    assert predicted[y == 0].mean() < 2 * FAR_TARGET
