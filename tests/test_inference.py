import numpy as np
import pytest

from photoquest_eval.inference import (
    CONFIRMED,
    NOT_FOUND,
    UNCERTAIN,
    ClassRule,
    Thresholds,
    presence_scores,
)


def test_presence_takes_max_confidence_per_target():
    names = {0: "bottle", 1: "cup", 2: "cell phone"}
    scores = presence_scores([0, 0, 1, 2], [0.3, 0.7, 0.9, 0.2], names, ["cell phone", "bottle"])
    assert scores.tolist() == pytest.approx([0.2, 0.7])


def test_presence_is_zero_without_detections():
    assert presence_scores([], [], {0: "bottle"}, ["bottle"]).tolist() == [0.0]


def test_three_statuses():
    thresholds = Thresholds("m", "r", 640, "confidence_95pct", {"bottle": ClassRule(0.5, 0.2)})
    assert thresholds.decide("bottle", 0.6) == CONFIRMED
    assert thresholds.decide("bottle", 0.5) == UNCERTAIN
    assert thresholds.decide("bottle", 0.2) == NOT_FOUND
    assert thresholds.decide("bottle", 0.0) == NOT_FOUND


def test_rule_order_is_checked():
    with pytest.raises(ValueError):
        ClassRule(high=0.1, low=0.5)


def test_thresholds_roundtrip(tmp_path):
    original = Thresholds("yolo11n-ft", "r1", 416, "confidence_95pct", {"tv": ClassRule(0.4)})
    path = tmp_path / "thresholds.json"
    original.to_json(path)
    assert Thresholds.from_json(path) == original


def test_best_boxes_keep_most_confident_detection():
    from photoquest_eval.inference import best_boxes

    names = {0: "bottle", 1: "cup"}
    scores, boxes = best_boxes(
        [0, 0, 1],
        [0.3, 0.8, 0.9],
        [[0, 0, 0.1, 0.1], [0.2, 0.2, 0.6, 0.7], [0, 0, 1, 1]],
        names,
        ["bottle", "tv"],
    )
    assert scores.tolist() == pytest.approx([0.8, 0.0])
    assert boxes[0].tolist() == pytest.approx([0.2, 0.2, 0.6, 0.7])
    assert np.isnan(boxes[1]).all()


def test_best_boxes_checks_lengths():
    from photoquest_eval.inference import best_boxes

    with pytest.raises(ValueError):
        best_boxes([0], [0.5], [], {0: "bottle"}, ["bottle"])
