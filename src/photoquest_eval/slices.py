"""Срезы оценки: «основная находка / фон» и мелкие объекты.

Рамки задаются в нормированных координатах ``(x_min, y_min, x_max, y_max)`` от 0 до 1,
чтобы одинаково обрабатывать COCO и Open Images.
"""

from __future__ import annotations

from collections.abc import Iterable

from photoquest_eval.config import MAIN_CENTER_ZONE, MAIN_MIN_AREA, SMALL_OBJECT_AREA

Box = tuple[float, float, float, float]

MAIN = "main"
BACKGROUND = "background"


def box_area(box: Box) -> float:
    x_min, y_min, x_max, y_max = box
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def is_main_object(
    box: Box,
    min_area: float = MAIN_MIN_AREA,
    center_zone: float = MAIN_CENTER_ZONE,
) -> bool:
    """Рамка достаточно крупная и её центр лежит в центральной зоне кадра."""
    x_min, y_min, x_max, y_max = box
    margin = (1 - center_zone) / 2
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    centered = margin <= center_x <= 1 - margin and margin <= center_y <= 1 - margin
    return box_area(box) >= min_area and centered


def positive_role(boxes: Iterable[Box]) -> str:
    """Роль класса на положительном фото: основная находка, если хоть одна рамка основная."""
    return MAIN if any(is_main_object(box) for box in boxes) else BACKGROUND


def is_small_object(box: Box, threshold: float = SMALL_OBJECT_AREA) -> bool:
    return box_area(box) < threshold


def coco_box_to_normalized(bbox: list[float], width: int, height: int) -> Box:
    """COCO ``[x, y, w, h]`` в пикселях → нормированные углы."""
    x, y, w, h = bbox
    return (x / width, y / height, (x + w) / width, (y + h) / height)
