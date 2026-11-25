import json

import numpy as np

from photoquest_eval.data import (
    NEGATIVE,
    POSITIVE,
    UNKNOWN,
    coco_to_yolo,
    copy_images,
    load_coco,
    load_openimages,
    sample_openimages,
    write_yolo_yaml,
)
from photoquest_eval.slices import (
    BACKGROUND,
    MAIN,
    coco_box_to_normalized,
    is_main_object,
    is_small_object,
    positive_role,
)


def test_main_object_rules():
    assert is_main_object((0.3, 0.3, 0.7, 0.7))
    assert not is_main_object((0.0, 0.0, 0.15, 0.15))  # мелко и в углу
    assert not is_main_object((0.0, 0.3, 0.3, 0.7))  # крупно, но у края
    assert positive_role([(0.0, 0.0, 0.1, 0.1), (0.3, 0.3, 0.7, 0.7)]) == MAIN
    assert positive_role([(0.0, 0.0, 0.1, 0.1)]) == BACKGROUND


def test_small_object_and_coco_box():
    box = coco_box_to_normalized([10, 20, 30, 40], 100, 200)
    assert box == (0.1, 0.1, 0.4, 0.3)
    assert is_small_object((0.0, 0.0, 0.05, 0.05))
    assert not is_small_object(box)


def _write_coco(path):
    coco = {
        "images": [
            {"id": 2, "file_name": "2.jpg", "width": 100, "height": 100},
            {"id": 1, "file_name": "1.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 44, "name": "bottle"}, {"id": 47, "name": "cup"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 44, "bbox": [30, 30, 40, 40], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 47, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        ],
    }
    path.write_text(json.dumps(coco), encoding="utf-8")


def test_load_coco(tmp_path):
    annotations = tmp_path / "instances.json"
    _write_coco(annotations)
    data = load_coco(annotations, tmp_path, targets=["bottle"])
    assert data.image_ids == ["1", "2"]
    assert data.labels[:, 0].tolist() == [POSITIVE, NEGATIVE]
    assert data.boxes[("1", "bottle")] == [(0.3, 0.3, 0.7, 0.7)]


def test_coco_to_yolo(tmp_path):
    annotations = tmp_path / "instances.json"
    _write_coco(annotations)
    for name in ("1.jpg", "2.jpg"):
        (tmp_path / name).write_bytes(b"")
    out = tmp_path / "out"
    stats = coco_to_yolo(
        annotations, tmp_path, out, ["bottle"], negative_fraction=1, val_fraction=0
    )
    assert stats["train"] == {"positive_images": 1, "negative_images": 1}
    assert stats["val"] == {"positive_images": 0, "negative_images": 0}
    label = (out / "train" / "labels" / "1.txt").read_text(encoding="utf-8")
    assert label == "0 0.500000 0.500000 0.400000 0.400000"
    assert (out / "train" / "labels" / "2.txt").read_text(encoding="utf-8") == ""
    assert (out / "train" / "images" / "1.jpg").is_symlink()
    yaml = write_yolo_yaml(out, ["bottle"]).read_text(encoding="utf-8")
    assert "train: train/images" in yaml and "0: bottle" in yaml


def test_coco_to_yolo_copies_and_resizes(tmp_path):
    from PIL import Image

    annotations = tmp_path / "instances.json"
    _write_coco(annotations)
    for name in ("1.jpg", "2.jpg"):
        Image.new("RGB", (100, 100), "red").save(tmp_path / name)
    out = tmp_path / "out"
    coco_to_yolo(annotations, tmp_path, out, ["bottle"], 1, 0, max_side=50, workers=2)
    copied = out / "train" / "images" / "1.jpg"
    assert not copied.is_symlink()
    with Image.open(copied) as image:
        assert image.size == (50, 50)
    # Повторный запуск не падает и пропускает готовые файлы.
    coco_to_yolo(annotations, tmp_path, out, ["bottle"], 1, 0, max_side=50, workers=2)


def test_copy_images_reports_failures(tmp_path):
    (tmp_path / "ok.jpg").write_bytes(b"jpeg")
    failed = copy_images([tmp_path / "ok.jpg", tmp_path / "missing.jpg"], tmp_path / "out")
    assert failed == [tmp_path / "missing.jpg"]
    assert (tmp_path / "out" / "ok.jpg").read_bytes() == b"jpeg"


def test_crowd_images_are_not_negatives(tmp_path):
    annotations = tmp_path / "instances.json"
    coco = {
        "images": [{"id": 1, "file_name": "1.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 44, "name": "bottle"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 44, "bbox": [0, 0, 5, 5], "iscrowd": 1}
        ],
    }
    annotations.write_text(json.dumps(coco), encoding="utf-8")
    (tmp_path / "1.jpg").write_bytes(b"")
    stats = coco_to_yolo(annotations, tmp_path, tmp_path / "out", ["bottle"], 1, 0)
    assert stats["train"] == {"positive_images": 0, "negative_images": 0}


def test_sample_openimages_is_deterministic(tmp_path):
    labels = tmp_path / "labels.csv"
    rows = ["ImageID,Source,LabelName,Confidence"]
    rows += [f"p{i},v,/m/04dr76w,1" for i in range(10)]
    rows += [f"n{i},v,/m/04dr76w,0" for i in range(10)]
    labels.write_text("\n".join(rows), encoding="utf-8")
    first = sample_openimages(labels, ["bottle"], positives_per_class=3, negatives_per_class=4)
    second = sample_openimages(labels, ["bottle"], positives_per_class=3, negatives_per_class=4)
    assert first == second
    assert sum(i.startswith("p") for i in first) == 3
    assert sum(i.startswith("n") for i in first) == 4


def test_load_openimages_uses_only_verified_labels(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for image_id in ("a", "b", "c", "d"):
        (images / f"{image_id}.jpg").write_bytes(b"")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "ImageID,Source,LabelName,Confidence\n"
        "a,verification,/m/04dr76w,1\n"  # бутылка есть
        "b,verification,/m/04dr76w,0\n"  # бутылки точно нет
        "c,verification,/m/0h8mzrc,0\n"  # нет настенных часов: про часы вообще неизвестно
        "d,verification,/m/046dlr,1\n",  # будильник — положительный для clock
        encoding="utf-8",
    )
    boxes = tmp_path / "boxes.csv"
    boxes.write_text(
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax\n"
        "a,xclick,/m/04dr76w,1,0.1,0.2,0.1,0.3\n",
        encoding="utf-8",
    )
    data = load_openimages(labels, boxes, images, targets=["bottle", "clock"])
    table = {image_id: data.labels[i].tolist() for i, image_id in enumerate(data.image_ids)}
    assert table["a"] == [POSITIVE, UNKNOWN]
    assert table["b"] == [NEGATIVE, UNKNOWN]
    assert "c" not in table  # ни для одного класса нет проверенной метки
    assert table["d"] == [UNKNOWN, POSITIVE]
    assert data.boxes[("a", "bottle")] == [(0.1, 0.1, 0.2, 0.3)]
    assert np.array_equal(data.mask(0), np.array([True, True, False]))
