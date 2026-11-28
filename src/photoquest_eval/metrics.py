"""Метрики присутствия класса и bootstrap-интервалы."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from photoquest_eval.config import SEED


def recall_far(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Recall по положительным и FAR по отрицательным примерам."""
    y = np.asarray(y).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    positives = int(y.sum())
    negatives = int((~y).sum())
    recall = float((predicted & y).sum() / positives) if positives else float("nan")
    far = float((predicted & ~y).sum() / negatives) if negatives else float("nan")
    return recall, far


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AP и ROC AUC по исходным score, без порога."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return {"average_precision": float("nan"), "roc_auc": float("nan")}
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def bootstrap_ci(
    y: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, tuple[float, float]]:
    """Перцентильные интервалы recall и FAR.

    Если заданы ``groups`` (например, источник или домохозяйство), ресэмплируются группы
    целиком, чтобы учесть сходство фото внутри группы.
    """
    y = np.asarray(y).astype(bool)
    predicted = np.asarray(predicted).astype(bool)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    recalls, fars = [], []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[g] for g in chosen])
        recall, far = recall_far(y[index], predicted[index])
        recalls.append(recall)
        fars.append(far)
    quantiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    return {
        "recall": tuple(float(v) for v in np.nanpercentile(recalls, quantiles)),
        "far": tuple(float(v) for v in np.nanpercentile(fars, quantiles)),
    }


def paired_bootstrap_diff(
    y: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float]]:
    """Разница recall и FAR моделей A − B на одних и тех же фото.

    Возвращает (оценка, нижняя граница, верхняя граница). Если интервал не содержит 0,
    различие считается значимым на уровне ``alpha``.
    """
    y = np.asarray(y).astype(bool)
    a = np.asarray(predicted_a).astype(bool)
    b = np.asarray(predicted_b).astype(bool)
    rng = np.random.default_rng(seed)
    point_a, point_b = recall_far(y, a), recall_far(y, b)
    diffs = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y), size=len(y))
        ra, fa = recall_far(y[index], a[index])
        rb, fb = recall_far(y[index], b[index])
        diffs.append((ra - rb, fa - fb))
    diffs = np.asarray(diffs)
    quantiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    low_r, high_r = np.nanpercentile(diffs[:, 0], quantiles)
    low_f, high_f = np.nanpercentile(diffs[:, 1], quantiles)
    return {
        "recall": (point_a[0] - point_b[0], float(low_r), float(high_r)),
        "far": (point_a[1] - point_b[1], float(low_f), float(high_f)),
    }
