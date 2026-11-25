"""Загрузка разметки COCO и Open Images в общий формат.

Общий формат для оценки: список ``image_ids`` и матрица меток ``labels`` (фото × классы)
со значениями 1 — класс есть, 0 — класса нет, -1 — неизвестно (фото не участвует
в оценке этого класса). Рамки возвращаются в нормированных координатах.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from photoquest_eval.config import HOME_CLASSES
from photoquest_eval.slices import Box, coco_box_to_normalized

POSITIVE, NEGATIVE, UNKNOWN = 1, 0, -1

# MID классов Open Images V7 (oidv7-class-descriptions-boxable.csv).
# Первый MID — сам класс, остальные — дочерние классы, которые считаются положительными.
# Отрицательной считается только проверенная метка основного класса.
OPENIMAGES_MIDS = {
    "Bottle": ("/m/04dr76w",),
    "Mobile phone": ("/m/050k8",),
    "Backpack": ("/m/01940j",),
    "Clock": ("/m/01x3z", "/m/046dlr", "/m/06_72j", "/m/0h8mzrc"),
    "Vase": ("/m/02s195",),
    "Houseplant": ("/m/03fp41",),
    "Laptop": ("/m/01c648",),
    "Computer keyboard": ("/m/01m2v",),
    "Television": ("/m/07c52",),
    "Banana": ("/m/09qck",),
}


@dataclass
class EvalSet:
    """Набор для оценки присутствия классов."""

    name: str
    targets: tuple[str, ...]
    image_ids: list[str]
    labels: np.ndarray
    boxes: dict[tuple[str, str], list[Box]] = field(default_factory=dict)
    image_paths: list[Path] = field(default_factory=list)

    def mask(self, column: int) -> np.ndarray:
        """Фото, для которых метка класса известна."""
        return self.labels[:, column] != UNKNOWN


def load_coco(
    annotation_path: Path,
    image_dir: Path,
    targets: Sequence[str] = tuple(HOME_CLASSES),
) -> EvalSet:
    """COCO: отсутствие метки класса считается отрицательным примером (как в ДЗ №2)."""
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = sorted(coco["images"], key=lambda row: row["id"])
    by_id = {row["id"]: row for row in images}
    categories = {row["id"]: row["name"] for row in coco["categories"]}
    missing = set(targets) - set(categories.values())
    if missing:
        raise ValueError(f"В COCO нет классов: {sorted(missing)}")
    column = {name: i for i, name in enumerate(targets)}
    row_of = {row["id"]: i for i, row in enumerate(images)}
    labels = np.zeros((len(images), len(targets)), dtype=np.int8)
    boxes: dict[tuple[str, str], list[Box]] = defaultdict(list)
    for ann in coco["annotations"]:
        name = categories[ann["category_id"]]
        if name not in column:
            continue
        labels[row_of[ann["image_id"]], column[name]] = POSITIVE
        if not ann.get("iscrowd", 0):
            image = by_id[ann["image_id"]]
            box = coco_box_to_normalized(ann["bbox"], image["width"], image["height"])
            boxes[(str(ann["image_id"]), name)].append(box)
    return EvalSet(
        name="coco",
        targets=tuple(targets),
        image_ids=[str(row["id"]) for row in images],
        labels=labels,
        boxes=dict(boxes),
        image_paths=[image_dir / row["file_name"] for row in images],
    )


def load_openimages(
    image_labels_csv: Path,
    boxes_csv: Path,
    image_dir: Path,
    targets: Sequence[str] = tuple(HOME_CLASSES),
) -> EvalSet:
    """Open Images: только проверенные человеком метки.

    Положительный пример — проверенная положительная метка или рамка класса.
    Отрицательный — проверенная отрицательная метка (Confidence 0) основного MID.
    Остальные фото для класса неизвестны и в его оценке не участвуют.
    Учитываются только фото, которые есть в ``image_dir``.
    """
    available = {path.stem: path for path in image_dir.glob("*.jpg")}
    mid_to_target: dict[str, str] = {}
    main_mid: dict[str, str] = {}
    for target in targets:
        mids = OPENIMAGES_MIDS[HOME_CLASSES[target]]
        main_mid[target] = mids[0]
        for mid in mids:
            mid_to_target[mid] = target
    status: dict[tuple[str, str], int] = {}
    with image_labels_csv.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target = mid_to_target.get(row["LabelName"])
            if target is None or row["ImageID"] not in available:
                continue
            key = (row["ImageID"], target)
            if float(row["Confidence"]) > 0:
                status[key] = POSITIVE
            elif row["LabelName"] == main_mid[target] and status.get(key) != POSITIVE:
                status[key] = NEGATIVE
    boxes: dict[tuple[str, str], list[Box]] = defaultdict(list)
    with boxes_csv.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target = mid_to_target.get(row["LabelName"])
            if target is None or row["ImageID"] not in available:
                continue
            key = (row["ImageID"], target)
            status[key] = POSITIVE
            boxes[key].append(
                (float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"]))
            )
    image_ids = sorted({image_id for image_id, _ in status})
    row_of = {image_id: i for i, image_id in enumerate(image_ids)}
    column = {name: i for i, name in enumerate(targets)}
    labels = np.full((len(image_ids), len(targets)), UNKNOWN, dtype=np.int8)
    for (image_id, target), value in status.items():
        labels[row_of[image_id], column[target]] = value
    return EvalSet(
        name="openimages",
        targets=tuple(targets),
        image_ids=image_ids,
        labels=labels,
        boxes=dict(boxes),
        image_paths=[available[image_id] for image_id in image_ids],
    )


def coco_to_yolo(
    annotation_path: Path,
    image_dir: Path,
    output_dir: Path,
    targets: Sequence[str] = tuple(HOME_CLASSES),
    negative_fraction: float = 0.1,
    val_fraction: float = 0.1,
    seed: int = 2026,
    max_side: int | None = None,
    workers: int = 32,
) -> dict[str, dict[str, int]]:
    """Готовит обучающую подвыборку COCO в формате YOLO: ``output_dir/{train,val}``.

    Берутся все фото с целевыми классами и случайная доля ``negative_fraction``
    фото без них, чтобы модель училась не срабатывать на посторонних сценах.
    Фото с crowd-разметкой целевого класса в негативы не попадают: предмет на них есть.
    Доля ``val_fraction`` фото откладывается для валидации при обучении.

    Без ``max_side`` создаются символические ссылки на исходные фото. С ``max_side``
    фото копируются в ``output_dir`` и уменьшаются до этого размера по длинной стороне:
    так обучение читает небольшие файлы с локального диска, а не с медленного
    сетевого хранилища. Метки YOLO нормированы и от размера фото не зависят.
    """
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {row["id"]: row["name"] for row in coco["categories"]}
    class_index = {name: i for i, name in enumerate(targets)}
    lines: dict[int, list[str]] = defaultdict(list)
    crowd_only: set[int] = set()
    images = {row["id"]: row for row in coco["images"]}
    for ann in coco["annotations"]:
        name = categories[ann["category_id"]]
        if name not in class_index:
            continue
        if ann.get("iscrowd", 0):
            crowd_only.add(ann["image_id"])
            continue
        image = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2) / image["width"]
        cy = (y + h / 2) / image["height"]
        lines[ann["image_id"]].append(
            f"{class_index[name]} {cx:.6f} {cy:.6f} {w / image['width']:.6f} "
            f"{h / image['height']:.6f}"
        )
    rng = np.random.default_rng(seed)
    negatives = np.array(sorted(set(images) - set(lines) - crowd_only), dtype=np.int64)
    keep = rng.random(len(negatives)) < negative_fraction
    selected = np.array(sorted(lines) + negatives[keep].tolist(), dtype=np.int64)
    is_val = rng.random(len(selected)) < val_fraction
    stats: dict[str, dict[str, int]] = {}
    for split, ids in (("train", selected[~is_val]), ("val", selected[is_val])):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        names = [images[image_id]["file_name"] for image_id in ids.tolist()]
        sources = [image_dir / name for name in names]
        targets_dir = output_dir / split / "images"
        if max_side is None:
            for source in sources:
                link = targets_dir / source.name
                if not link.exists():
                    link.symlink_to(source.resolve())
        else:
            failed = copy_images(sources, targets_dir, max_side=max_side, workers=workers)
            if failed:
                raise OSError(f"Не удалось скопировать {len(failed)} фото, например {failed[0]}")
        positives = 0
        for image_id, name in zip(ids.tolist(), names):  # noqa: B905 — длины совпадают
            label = output_dir / split / "labels" / Path(name).with_suffix(".txt").name
            label.write_text("\n".join(lines.get(image_id, [])), encoding="utf-8")
            positives += image_id in lines
        stats[split] = {"positive_images": positives, "negative_images": len(ids) - positives}
    return stats


def copy_images(
    sources: Sequence[Path],
    output_dir: Path,
    max_side: int | None = None,
    workers: int = 32,
) -> list[Path]:
    """Параллельно копирует фото в ``output_dir``, при ``max_side`` уменьшая их.

    Уже скопированные файлы пропускаются, поэтому прерванное копирование можно
    продолжить повторным вызовом. Возвращает пути, которые скопировать не удалось.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    output_dir.mkdir(parents=True, exist_ok=True)

    def copy_one(source: Path) -> Path | None:
        target = output_dir / source.name
        if target.exists() and target.stat().st_size:
            return None
        temporary = target.with_suffix(".part")
        try:
            if max_side is None:
                shutil.copyfile(source, temporary)
            else:
                from PIL import Image

                with Image.open(source) as image:
                    image = image.convert("RGB")
                    image.thumbnail((max_side, max_side))
                    image.save(temporary, format="JPEG", quality=95)
            temporary.replace(target)
        except OSError:
            return source
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [path for path in pool.map(copy_one, sources) if path is not None]


def write_yolo_yaml(dataset_dir: Path, targets: Sequence[str] = tuple(HOME_CLASSES)) -> Path:
    """Конфиг датасета Ultralytics для каталога из ``coco_to_yolo``."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(targets))
    path = dataset_dir / "dataset.yaml"
    path.write_text(
        f"path: {dataset_dir.resolve()}\ntrain: train/images\nval: val/images\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return path


OPENIMAGES_URLS = {
    "image_labels": (
        "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv"
    ),
    "boxes": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "image": "https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg",
}


def sample_openimages(
    image_labels_csv: Path,
    targets: Sequence[str] = tuple(HOME_CLASSES),
    positives_per_class: int = 500,
    negatives_per_class: int = 1000,
    seed: int = 2026,
) -> list[str]:
    """Фиксированная выборка фото Open Images test до запуска моделей.

    Для каждого класса берётся до ``positives_per_class`` фото с проверенной
    положительной меткой и до ``negatives_per_class`` с проверенной отрицательной.
    """
    positives: dict[str, set[str]] = defaultdict(set)
    negatives: dict[str, set[str]] = defaultdict(set)
    mid_to_target = {}
    main_mid = {}
    for target in targets:
        mids = OPENIMAGES_MIDS[HOME_CLASSES[target]]
        main_mid[target] = mids[0]
        for mid in mids:
            mid_to_target[mid] = target
    with image_labels_csv.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target = mid_to_target.get(row["LabelName"])
            if target is None:
                continue
            if float(row["Confidence"]) > 0:
                positives[target].add(row["ImageID"])
            elif row["LabelName"] == main_mid[target]:
                negatives[target].add(row["ImageID"])
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for target in targets:
        for pool, limit in (
            (positives[target], positives_per_class),
            (negatives[target] - positives[target], negatives_per_class),
        ):
            ordered = sorted(pool)
            take = min(limit, len(ordered))
            chosen.update(ordered[i] for i in rng.choice(len(ordered), take, replace=False))
    return sorted(chosen)


def download_openimages(
    image_ids: Sequence[str],
    image_dir: Path,
    max_side: int = 1024,
    workers: int = 32,
) -> list[str]:
    """Скачивает фото Open Images test. Возвращает ID, которые скачать не удалось.

    Оригиналы весят мегабайты, поэтому фото уменьшаются до ``max_side`` по длинной
    стороне. Рамки Open Images нормированы и от размера не зависят.
    """
    import io
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    image_dir.mkdir(parents=True, exist_ok=True)

    def fetch(image_id: str) -> str | None:
        path = image_dir / f"{image_id}.jpg"
        if path.exists() and path.stat().st_size:
            return None
        try:
            url = OPENIMAGES_URLS["image"].format(image_id=image_id)
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert("RGB")
                image.thumbnail((max_side, max_side))
                temporary = path.with_suffix(".part")
                image.save(temporary, format="JPEG", quality=90)
            temporary.replace(path)
        except OSError:
            return image_id
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [image_id for image_id in pool.map(fetch, image_ids) if image_id]
