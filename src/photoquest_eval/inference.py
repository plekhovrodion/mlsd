"""Инференс проверки фото: от детекций к статусу задания.

Этот модуль — эталон логики, которую в ДЗ №4 повторит код в браузере.
Функции без зависимостей от torch, кроме ``predict_scores``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

CONFIRMED = "confirmed"
UNCERTAIN = "uncertain"
NOT_FOUND = "not_found"


def best_boxes(
    class_ids: Iterable[int],
    confidences: Iterable[float],
    boxes: Iterable[Sequence[float]],
    names: dict[int, str] | Sequence[str],
    targets: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Score и рамка самой уверенной детекции каждого класса.

    Возвращает score (классы,) — 0 без детекций — и рамки (классы, 4) в нормированных
    координатах ``x1, y1, x2, y2``; у классов без детекций рамка из NaN.
    """
    column = {name: i for i, name in enumerate(targets)}
    class_ids = np.asarray(list(class_ids), dtype=int)
    confidences = np.asarray(list(confidences), dtype=np.float32)
    boxes = np.asarray(list(boxes), dtype=np.float32).reshape(-1, 4)
    if not class_ids.shape == confidences.shape == (len(boxes),):
        raise ValueError("Число классов, уверенностей и рамок детекций не совпадает")
    scores = np.zeros(len(targets), dtype=np.float32)
    best = np.full((len(targets), 4), np.nan, dtype=np.float32)
    for class_id, confidence, box in zip(class_ids, confidences, boxes):  # noqa: B905
        index = column.get(names[int(class_id)])
        if index is not None and confidence > scores[index]:
            scores[index] = confidence
            best[index] = box
    return scores, best


def presence_scores(
    class_ids: Iterable[int],
    confidences: Iterable[float],
    names: dict[int, str] | Sequence[str],
    targets: Sequence[str],
) -> np.ndarray:
    """Score присутствия класса — максимальная уверенность его детекций, иначе 0."""
    class_ids = list(class_ids)
    confidences = list(confidences)
    if len(class_ids) != len(confidences):
        raise ValueError("Число классов и уверенностей детекций не совпадает")
    return best_boxes(class_ids, confidences, [[0, 0, 0, 0]] * len(class_ids), names, targets)[0]


@dataclass(frozen=True)
class ClassRule:
    """Порог класса. ``score > high`` — верно, ``score <= low`` — не найдено."""

    high: float
    low: float = 0.0

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError(f"low={self.low} > high={self.high}")


@dataclass(frozen=True)
class Thresholds:
    """Версионированный набор порогов, который загружает игра."""

    model_version: str
    rule_version: str
    image_size: int
    protocol: str
    rules: dict[str, ClassRule] = field(default_factory=dict)

    def decide(self, target: str, score: float) -> str:
        rule = self.rules[target]
        if score > rule.high:
            return CONFIRMED
        if score <= rule.low:
            return NOT_FOUND
        return UNCERTAIN

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Thresholds:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rules = {name: ClassRule(**rule) for name, rule in payload.pop("rules").items()}
        return cls(rules=rules, **payload)


def predict_best_boxes(
    model_path: str | Path,
    image_paths: Sequence[str | Path],
    targets: Sequence[str],
    image_size: int,
    conf: float,
    iou: float,
    max_det: int,
    batch_size: int = 32,
    device: int | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Прогоняет YOLO по фото: score (фото × классы) и лучшие рамки (фото × классы × 4)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    scores = np.zeros((len(image_paths), len(targets)), dtype=np.float32)
    boxes = np.full((len(image_paths), len(targets), 4), np.nan, dtype=np.float32)
    for offset in range(0, len(image_paths), batch_size):
        batch = [str(p) for p in image_paths[offset : offset + batch_size]]
        results = model.predict(
            source=batch,
            imgsz=image_size,
            conf=conf,
            iou=iou,
            max_det=max_det,
            batch=len(batch),
            device=device,
            verbose=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(f"YOLO вернул {len(results)} ответов на {len(batch)} фото")
        for row, result in enumerate(results, start=offset):
            if result.boxes is None:
                continue
            scores[row], boxes[row] = best_boxes(
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxyn.cpu().numpy(),
                model.names,
                targets,
            )
    return scores, boxes


def predict_scores(
    model_path: str | Path,
    image_paths: Sequence[str | Path],
    targets: Sequence[str],
    image_size: int,
    conf: float,
    iou: float,
    max_det: int,
    batch_size: int = 32,
    device: int | str | None = None,
) -> np.ndarray:
    """Прогоняет YOLO по фото и возвращает матрицу score (фото × классы)."""
    return predict_best_boxes(
        model_path, image_paths, targets, image_size, conf, iou, max_det, batch_size, device
    )[0]
