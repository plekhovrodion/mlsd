"""Компромисс recall / FAR на трудных негативах.

Порог, подобранный на негативах COCO, почти не защищает от похожих предметов
(банка вместо бутылки). Здесь порог подбирается на проверенных негативах Open Images:
фото делятся на два фолда, порог считается на одном и проверяется на другом,
затем фолды меняются местами — оценка не подглядывает в проверочные фото.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from photoquest_eval.calibration import threshold_for_far


@dataclass(frozen=True)
class TradeoffPoint:
    """Точка компромисса: целевой FAR на трудных негативах и результат на отложенном фолде."""

    level: float
    cutoffs: tuple[float, float]
    recall: float
    far: float


def two_fold_tradeoff(
    labels: np.ndarray,
    scores: np.ndarray,
    floor: float,
    levels: Sequence[float],
    fold: np.ndarray,
) -> list[TradeoffPoint]:
    """Порог для каждого уровня FAR на трудных негативах, оценённый на другом фолде.

    ``labels`` — 1, 0 или -1 (неизвестно, не участвует). Порог не опускается ниже
    ``floor`` (например, порога COCO), чтобы на обычных фото FAR не вырос.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    fold = np.asarray(fold, dtype=bool)
    positive, negative = labels == 1, labels == 0
    points = []
    for level in levels:
        cutoffs, true_pos, false_pos = [], 0, 0
        for calib in (fold, ~fold):
            test = ~calib
            cut = max(floor, threshold_for_far(scores[negative & calib], "empirical_5pct", level))
            cutoffs.append(cut)
            true_pos += int((scores[positive & test] > cut).sum())
            false_pos += int((scores[negative & test] > cut).sum())
        points.append(
            TradeoffPoint(
                level=float(level),
                cutoffs=(cutoffs[0], cutoffs[1]),
                recall=true_pos / max(int(positive.sum()), 1),
                far=false_pos / max(int(negative.sum()), 1),
            )
        )
    return points


def mean_rates(
    labels: np.ndarray, scores: np.ndarray, cutoffs: Sequence[float]
) -> tuple[float, float]:
    """Recall и FAR, усреднённые по нескольким порогам (например, порогам двух фолдов)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    positive, negative = scores[labels == 1], scores[labels == 0]
    recall = float(np.mean([(positive > cut).mean() for cut in cutoffs])) if len(positive) else 0.0
    far = float(np.mean([(negative > cut).mean() for cut in cutoffs])) if len(negative) else 0.0
    return recall, far


def status_rates(scores: np.ndarray, low: float, high: float) -> tuple[float, float, float]:
    """Доли статусов «да» (> high), «не уверен» (low, high] и «не найдено» (<= low)."""
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores):
        return 0.0, 0.0, 0.0
    confirmed = float((scores > high).mean())
    not_found = float((scores <= low).mean())
    return confirmed, 1.0 - confirmed - not_found, not_found
