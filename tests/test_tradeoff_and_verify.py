import math

import numpy as np
import pytest

from photoquest_eval.data import boxable_image_ids
from photoquest_eval.tradeoff import mean_rates, status_rates, two_fold_tradeoff
from photoquest_eval.verify import LOOKALIKES, PROMPTS, crop_box, prompts, target_probability


def test_two_fold_tradeoff_respects_floor_and_is_held_out():
    # Позитивы выше негативов: при FAR 0 порог отсекает все негативы и сохраняет позитивы.
    labels = np.array([1] * 6 + [0] * 6 + [-1] * 2)
    scores = np.array([0.9] * 6 + [0.1, 0.2, 0.3, 0.1, 0.2, 0.3] + [0.95, 0.0])
    fold = np.array([True, False] * 7)
    (point,) = two_fold_tradeoff(labels, scores, floor=0.0, levels=[0.0], fold=fold)
    assert point.recall == 1.0 and point.far == 0.0
    assert all(cut >= 0.3 for cut in point.cutoffs)
    (floored,) = two_fold_tradeoff(labels, scores, floor=0.95, levels=[0.5], fold=fold)
    assert floored.cutoffs == (0.95, 0.95) and floored.recall == 0.0


def test_mean_rates_and_status_rates():
    labels = np.array([1, 1, 0, 0, -1])
    scores = np.array([0.9, 0.4, 0.5, 0.1, 1.0])
    assert mean_rates(labels, scores, [0.3, 0.6]) == pytest.approx((0.75, 0.25))
    assert status_rates(np.array([0.1, 0.5, 0.9, 0.95]), low=0.2, high=0.8) == pytest.approx(
        (0.5, 0.25, 0.25)
    )
    assert status_rates(np.array([]), 0.1, 0.2) == (0.0, 0.0, 0.0)


def test_prompts_cover_all_targets_and_start_with_class():
    assert set(PROMPTS) == set(LOOKALIKES)
    texts = prompts("bottle")
    assert texts[0] == "a photo of a bottle." and "a photo of a can." in texts
    assert len(texts) == 1 + len(LOOKALIKES["bottle"])


def test_crop_box_margins_clamp_and_nan():
    assert crop_box(100, 200, (0.4, 0.4, 0.6, 0.6), margin=0.0, min_side=1) == (40, 80, 60, 120)
    assert crop_box(100, 100, (0.0, 0.0, 1.0, 1.0), margin=0.1) == (0, 0, 100, 100)
    # Маленькая рамка расширяется до min_side вокруг центра.
    tiny = crop_box(1000, 1000, (0.5, 0.5, 0.501, 0.501), margin=0, min_side=32)
    assert tiny == (484, 484, 517, 517)
    assert crop_box(100, 100, (math.nan,) * 4) is None


def test_target_probability_is_softmax_of_first():
    sims = np.array([[0.3, 0.2, 0.1], [0.1, 0.1, 0.1]])
    p = target_probability(sims, scale=10.0)
    expected = np.exp(3) / (np.exp(3) + np.exp(2) + np.exp(1))
    assert p[0] == pytest.approx(expected)
    assert p[1] == pytest.approx(1 / 3)


def test_boxable_image_ids(tmp_path):
    boxes = tmp_path / "boxes.csv"
    boxes.write_text("ImageID,Source,LabelName\na,x,/m/1\na,x,/m/2\nb,x,/m/1\n", encoding="utf-8")
    assert boxable_image_ids(boxes) == {"a", "b"}
