"""Вторая ступень проверки: вырез лучшей рамки детектора сравнивается с подписями.

Детектор видит предмет на кадре 640 px и плохо отличает его от похожих (банка
вместо бутылки). Здесь вырез рамки оценивает MobileCLIP: подпись целевого класса
против подписей похожих предметов. Списки похожих предметов зафиксированы
до запуска на Open Images и не подбирались по результатам.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PROMPTS = {
    "bottle": "a bottle",
    "cell phone": "a mobile phone",
    "backpack": "a backpack",
    "clock": "a clock",
    "vase": "a vase",
    "potted plant": "a potted plant",
    "laptop": "a laptop",
    "keyboard": "a computer keyboard",
    "tv": "a television",
    "banana": "a banana",
}

LOOKALIKES = {
    "bottle": ("a can", "a jar", "a cup", "a glass"),
    "cell phone": ("a remote control", "a tablet computer", "a calculator", "a wallet"),
    "backpack": ("a handbag", "a suitcase", "a jacket", "a shopping bag"),
    "clock": ("a plate", "a wristwatch", "a compass", "a round sign"),
    "vase": ("a jar", "a bottle", "a flower pot", "a cup"),
    "potted plant": ("a bouquet of flowers", "a tree", "a vase", "a garden bush"),
    "laptop": ("a tablet computer", "a computer monitor", "a book", "a computer keyboard"),
    "keyboard": ("a laptop", "a piano keyboard", "a remote control", "a calculator"),
    "tv": ("a computer monitor", "a laptop", "a picture frame", "a window"),
    "banana": ("a lemon", "a mango", "a corn cob", "a yellow pepper"),
}

# Масштаб логитов CLIP: сходство косинусов умножается на него перед softmax.
LOGIT_SCALE = 100.0


def prompts(target: str) -> list[str]:
    """Подписи для класса: первая — сам класс, остальные — похожие предметы."""
    return [f"a photo of {text}." for text in (PROMPTS[target], *LOOKALIKES[target])]


def crop_box(
    width: int, height: int, box: Sequence[float], margin: float = 0.1, min_side: int = 32
) -> tuple[int, int, int, int] | None:
    """Пиксельные координаты выреза: рамка с полями ``margin`` от её размера, в пределах кадра.

    ``box`` — нормированные ``x1, y1, x2, y2``. Возвращает ``None`` для рамки из NaN.
    Слишком маленький вырез расширяется до ``min_side`` пикселей вокруг центра.
    """
    if box is None or not np.all(np.isfinite(box)):
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = (x2 - x1) * width, (y2 - y1) * height
    cx, cy = (x1 + x2) / 2 * width, (y1 + y2) / 2 * height
    half_w = max(w * (1 + 2 * margin), min_side) / 2
    half_h = max(h * (1 + 2 * margin), min_side) / 2
    left = int(max(0, np.floor(cx - half_w)))
    top = int(max(0, np.floor(cy - half_h)))
    right = int(min(width, np.ceil(cx + half_w)))
    bottom = int(min(height, np.ceil(cy + half_h)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def target_probability(similarities: np.ndarray, scale: float = LOGIT_SCALE) -> np.ndarray:
    """Вероятность первой подписи (класса) среди всех подписей по косинусным сходствам."""
    logits = scale * np.asarray(similarities, dtype=np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights[..., 0] / weights.sum(axis=-1)
