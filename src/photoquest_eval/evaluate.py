"""Сборка оценки: пороги, метрики по классам, срезы и интервалы.

Функции возвращают списки словарей, чтобы ноутбук собирал из них таблицы pandas.
"""

from __future__ import annotations

import numpy as np

from photoquest_eval.calibration import evaluate_class_oof, threshold_for_far
from photoquest_eval.config import FAR_TARGET, PROTOCOLS
from photoquest_eval.data import POSITIVE, EvalSet
from photoquest_eval.inference import ClassRule, Thresholds
from photoquest_eval.metrics import bootstrap_ci, ranking_metrics, recall_far
from photoquest_eval.slices import BACKGROUND, MAIN, is_small_object, positive_role

UNKNOWN_ROLE = "no_box"


def positive_roles(eval_set: EvalSet, column: int) -> np.ndarray:
    """Роль класса на каждом фото: main, background или no_box (рамки нет)."""
    target = eval_set.targets[column]
    roles = np.full(len(eval_set.image_ids), "", dtype=object)
    for row, image_id in enumerate(eval_set.image_ids):
        if eval_set.labels[row, column] != POSITIVE:
            continue
        boxes = eval_set.boxes.get((image_id, target))
        roles[row] = positive_role(boxes) if boxes else UNKNOWN_ROLE
    return roles


def has_only_small_objects(eval_set: EvalSet, column: int) -> np.ndarray:
    """Положительное фото, где все рамки класса меньше 1% кадра."""
    target = eval_set.targets[column]
    result = np.zeros(len(eval_set.image_ids), dtype=bool)
    for row, image_id in enumerate(eval_set.image_ids):
        boxes = eval_set.boxes.get((image_id, target))
        if boxes:
            result[row] = all(is_small_object(box) for box in boxes)
    return result


def slice_recalls(eval_set: EvalSet, column: int, predicted: np.ndarray) -> dict[str, float]:
    """Recall в срезах положительных фото."""
    y = eval_set.labels[:, column] == POSITIVE
    roles = positive_roles(eval_set, column)
    small = has_only_small_objects(eval_set, column)
    result: dict[str, float] = {}
    for name, mask in (
        ("main", y & (roles == MAIN)),
        ("background", y & (roles == BACKGROUND)),
        ("small_only", y & small),
    ):
        result[f"positives_{name}"] = int(mask.sum())
        result[f"recall_{name}"] = float(predicted[mask].mean()) if mask.any() else float("nan")
    return result


def coco_oof(
    eval_set: EvalSet, scores: np.ndarray, model: str, image_size: int
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    """Протокол ДЗ №2: пороги на четырёх фолдах, оценка на пятом."""
    rows, calibration_rows, predictions = [], [], {}
    for column, target in enumerate(eval_set.targets):
        y = eval_set.labels[:, column].astype(int)
        values = scores[:, column]
        for protocol in PROTOCOLS:
            predicted, folds, calibration = evaluate_class_oof(y, values, protocol)
            recall, far = recall_far(y, predicted)
            ci = bootstrap_ci(y, predicted)
            rows.append(
                {
                    "dataset": eval_set.name,
                    "model": model,
                    "image_size": image_size,
                    "target": target,
                    "protocol": protocol,
                    "positive_images": int(y.sum()),
                    "negative_images": int((y == 0).sum()),
                    **ranking_metrics(y, values),
                    "recall": recall,
                    "recall_low": ci["recall"][0],
                    "recall_high": ci["recall"][1],
                    "far": far,
                    "far_low": ci["far"][0],
                    "far_high": ci["far"][1],
                    "far_exceeds_target": far > FAR_TARGET,
                    **slice_recalls(eval_set, column, predicted),
                }
            )
            for row in calibration:
                calibration_rows.append(
                    {"model": model, "image_size": image_size, "target": target, **row}
                )
            predictions[f"{target}__{protocol}"] = predicted
            predictions[f"{target}__fold"] = folds
    return rows, calibration_rows, predictions


def fit_thresholds(
    eval_set: EvalSet,
    scores: np.ndarray,
    model_version: str,
    image_size: int,
    rule_version: str,
    protocol: str = "confidence_95pct",
) -> Thresholds:
    """Итоговые пороги по всем известным негативам набора (для переноса и для игры)."""
    rules = {}
    for column, target in enumerate(eval_set.targets):
        known = eval_set.mask(column)
        negatives = scores[known & (eval_set.labels[:, column] == 0), column]
        rules[target] = ClassRule(high=threshold_for_far(negatives, protocol))
    return Thresholds(model_version, rule_version, image_size, protocol, rules)


def transfer_metrics(
    eval_set: EvalSet,
    scores: np.ndarray,
    thresholds: Thresholds,
    model: str,
    n_resamples: int = 1000,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Оценка на другом наборе с порогами, зафиксированными заранее."""
    rows, predictions = [], {}
    for column, target in enumerate(eval_set.targets):
        known = eval_set.mask(column)
        y = (eval_set.labels[:, column] == POSITIVE).astype(int)
        values = scores[:, column]
        predicted = values > thresholds.rules[target].high
        recall, far = recall_far(y[known], predicted[known])
        ci = bootstrap_ci(y[known], predicted[known], n_resamples=n_resamples)
        rows.append(
            {
                "dataset": eval_set.name,
                "model": model,
                "image_size": thresholds.image_size,
                "target": target,
                "protocol": thresholds.protocol,
                "cutoff": thresholds.rules[target].high,
                "positive_images": int(y[known].sum()),
                "negative_images": int((known & (y == 0)).sum()),
                **ranking_metrics(y[known], values[known]),
                "recall": recall,
                "recall_low": ci["recall"][0],
                "recall_high": ci["recall"][1],
                "far": far,
                "far_low": ci["far"][0],
                "far_high": ci["far"][1],
                "far_exceeds_target": far > FAR_TARGET,
                **slice_recalls(eval_set, column, predicted & known),
            }
        )
        predictions[target] = predicted & known
    return rows, predictions
