"""Калибровка порогов по отрицательным примерам и out-of-fold оценка."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom
from sklearn.model_selection import StratifiedKFold

from photoquest_eval.config import CONFIDENCE, CV_SPLITS, FAR_TARGET, SEED


def threshold_for_far(
    negative_scores: np.ndarray,
    protocol: str = "empirical_5pct",
    far_target: float = FAR_TARGET,
    confidence: float = CONFIDENCE,
) -> float:
    """Порог для строгого сравнения ``score > cutoff``.

    ``empirical_5pct`` допускает долю ``far_target`` ошибок на калибровочных негативах.
    ``confidence_95pct`` уменьшает допустимое число ошибок так, чтобы истинный FAR
    не превышал цель с уровнем доверия ``confidence`` (односторонняя биномиальная граница).
    """
    negatives = np.asarray(negative_scores, dtype=np.float64)
    if not negatives.size or not np.isfinite(negatives).all():
        raise ValueError("Нужны конечные score отрицательных примеров")
    count = len(negatives)
    allowed_fp = int(math.floor(far_target * count))
    if protocol == "confidence_95pct":
        candidates = np.arange(allowed_fp + 1)
        valid = candidates[binom.cdf(candidates, count, far_target) <= 1 - confidence]
        if not len(valid):
            return math.inf
        allowed_fp = int(valid[-1])
    elif protocol != "empirical_5pct":
        raise ValueError(f"Неизвестный протокол: {protocol}")
    cutoff = float(np.sort(negatives)[::-1][allowed_fp])
    # Без nextafter: эпсилон float64 может исчезнуть при сравнении во float32.
    assert np.mean(negatives > cutoff) <= far_target
    return cutoff


def evaluate_class_oof(
    y: np.ndarray,
    scores: np.ndarray,
    protocol: str,
    n_splits: int = CV_SPLITS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Пороги на четырёх фолдах, применение к пятому.

    Возвращает предсказания out-of-fold, номер фолда каждого фото и записи калибровки.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predicted = np.zeros(len(y), dtype=bool)
    folds = np.full(len(y), -1, dtype=int)
    calibration = []
    for fold, (calibration_idx, test_idx) in enumerate(cv.split(scores, y)):
        negatives = scores[calibration_idx][y[calibration_idx] == 0]
        cutoff = threshold_for_far(negatives, protocol)
        predicted[test_idx] = scores[test_idx] > cutoff
        folds[test_idx] = fold
        calibration.append(
            {
                "protocol": protocol,
                "fold": fold,
                "cutoff": cutoff,
                "calibration_far": float(np.mean(negatives > cutoff)),
                "calibration_negatives": len(negatives),
                "test_positives": int(y[test_idx].sum()),
            }
        )
    assert (folds >= 0).all()
    return predicted, folds, calibration
