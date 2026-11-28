import numpy as np

from photoquest_eval.data import NEGATIVE, POSITIVE, UNKNOWN, EvalSet
from photoquest_eval.evaluate import coco_oof, fit_thresholds, positive_roles, transfer_metrics


def _eval_set(labels, boxes=None):
    labels = np.asarray(labels, dtype=np.int8).reshape(-1, 1)
    return EvalSet(
        name="test",
        targets=("bottle",),
        image_ids=[str(i) for i in range(len(labels))],
        labels=labels,
        boxes=boxes or {},
    )


def test_positive_roles():
    data = _eval_set(
        [POSITIVE, POSITIVE, POSITIVE, NEGATIVE],
        {("0", "bottle"): [(0.3, 0.3, 0.7, 0.7)], ("1", "bottle"): [(0.0, 0.0, 0.1, 0.1)]},
    )
    assert positive_roles(data, 0).tolist() == ["main", "background", "no_box", ""]


def test_thresholds_ignore_unknown_labels():
    labels = [NEGATIVE] * 100 + [UNKNOWN] * 10
    scores = np.concatenate([np.linspace(0, 0.5, 100), np.ones(10)]).reshape(-1, 1)
    thresholds = fit_thresholds(_eval_set(labels), scores, "m", 640, "r1", "empirical_5pct")
    assert thresholds.rules["bottle"].high < 0.5


def test_transfer_metrics_on_known_labels_only():
    labels = [POSITIVE] * 20 + [NEGATIVE] * 80 + [UNKNOWN] * 50
    scores = np.concatenate([np.full(20, 0.9), np.full(80, 0.1), np.full(50, 0.9)])
    data = _eval_set(labels)
    thresholds = fit_thresholds(data, scores.reshape(-1, 1), "m", 640, "r1", "empirical_5pct")
    rows, predictions = transfer_metrics(data, scores.reshape(-1, 1), thresholds, "m", 50)
    assert rows[0]["recall"] == 1.0
    assert rows[0]["far"] == 0.0
    assert rows[0]["positive_images"] == 20 and rows[0]["negative_images"] == 80
    assert not predictions["bottle"][100:].any()


def test_coco_oof_rows():
    rng = np.random.default_rng(0)
    labels = (rng.random(400) < 0.25).astype(int)
    scores = np.where(labels == 1, 0.8, 0.2) + rng.normal(0, 0.05, 400)
    rows, calibration, predictions = coco_oof(_eval_set(labels), scores.reshape(-1, 1), "m", 640)
    assert len(rows) == 2 and len(calibration) == 10
    assert {"bottle__empirical_5pct", "bottle__confidence_95pct"} <= set(predictions)
    assert rows[0]["recall_low"] <= rows[0]["recall"] <= rows[0]["recall_high"]
