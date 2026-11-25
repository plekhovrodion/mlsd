import numpy as np
import pytest

from photoquest_eval.metrics import (
    bootstrap_ci,
    paired_bootstrap_diff,
    ranking_metrics,
    recall_far,
)


def test_recall_far():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    predicted = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert recall_far(y, predicted) == pytest.approx((0.75, 1 / 6))


def test_ranking_metrics_single_class_is_nan():
    result = ranking_metrics(np.ones(5), np.arange(5))
    assert np.isnan(result["average_precision"])


def test_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.2
    predicted = np.where(y, rng.random(2000) < 0.8, rng.random(2000) < 0.05)
    recall, far = recall_far(y, predicted)
    ci = bootstrap_ci(y, predicted, n_resamples=300)
    assert ci["recall"][0] <= recall <= ci["recall"][1]
    assert ci["far"][0] <= far <= ci["far"][1]


def test_group_bootstrap_runs():
    y = np.array([1, 0] * 50)
    predicted = y.copy()
    ci = bootstrap_ci(y, predicted, groups=np.repeat(np.arange(10), 10), n_resamples=50)
    assert ci["recall"] == (1.0, 1.0)


def test_paired_diff_identical_models_is_zero():
    y = np.array([1, 0] * 100)
    predicted = np.array([1, 1, 0, 0] * 50)
    diff = paired_bootstrap_diff(y, predicted, predicted, n_resamples=100)
    assert diff["recall"] == (0.0, 0.0, 0.0)
