from photoquest_eval.data import (
    NEGATIVE,
    POSITIVE,
    openimages_to_yolo,
    sample_hard_negatives,
    scan_openimages_boxes,
    scan_openimages_labels,
)

BOTTLE, PHONE, ALARM = "/m/04dr76w", "/m/050k8", "/m/046dlr"


def _labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "ImageID,Source,LabelName,Confidence\n"
        f"can,verification,{BOTTLE},0.0\n"  # банка: проверенно не бутылка
        f"can,verification,{PHONE},1.0\n"  # но телефон на фото есть
        f"jar,verification,{BOTTLE},0\n"
        f"pos,verification,{BOTTLE},1\n"
        f"pos,verification,/m/0zzz,1\n"  # посторонний класс игнорируется
        f"alarm,verification,{ALARM},1\n",  # дочерний MID часов
        encoding="utf-8",
    )
    return path


def test_scan_labels(tmp_path):
    status = scan_openimages_labels(_labels(tmp_path), ["bottle", "cell phone", "clock"])
    assert status == {
        "can": {"bottle": NEGATIVE, "cell phone": POSITIVE},
        "jar": {"bottle": NEGATIVE},
        "pos": {"bottle": POSITIVE},
        "alarm": {"clock": POSITIVE},
    }


def test_sample_is_deterministic_and_limited():
    status = {f"n{i}": {"bottle": NEGATIVE} for i in range(10)}
    status.update({f"p{i}": {"bottle": POSITIVE} for i in range(10)})
    first = sample_hard_negatives(status, ["bottle"], positives_per_class=2, negatives_per_class=3)
    assert first == sample_hard_negatives(status, ["bottle"], 2, 3)
    assert sum(i.startswith("n") for i in first) == 3
    assert sum(i.startswith("p") for i in first) == 2


def test_scan_boxes_and_yolo(tmp_path):
    targets = ["bottle", "cell phone"]
    status = scan_openimages_labels(_labels(tmp_path), targets)
    boxes_csv = tmp_path / "boxes.csv"
    boxes_csv.write_text(
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsGroupOf\n"
        f"can,xclick,{PHONE},1,0.1,0.3,0.2,0.6,0\n"
        f"other,xclick,{PHONE},1,0,1,0,1,0\n",  # фото не выбрано
        encoding="utf-8",
    )
    ids = ["can", "jar", "pos", "lost"]
    boxes = scan_openimages_boxes(boxes_csv, ids, targets)
    assert boxes == {("can", "cell phone"): [(0.1, 0.2, 0.3, 0.6)]}

    images = tmp_path / "images"
    images.mkdir()
    for image_id in ("can", "jar", "pos"):
        (images / f"{image_id}.jpg").write_bytes(b"jpeg")
    out = tmp_path / "yolo"
    stats = openimages_to_yolo(ids, status, boxes, images, out, targets, val_fraction=0)
    assert stats["train"] == {
        "positive_images": 1,  # can: телефон с рамкой
        "negative_images": 1,  # jar: трудный негатив без предметов
        "no_boxes": 1,  # pos: бутылка без рамки — пропуск, иначе она стала бы фоном
        "missing": 1,  # lost: фото не скачалось
    }
    label = (out / "train" / "labels" / "oi_can.txt").read_text(encoding="utf-8")
    assert label == "1 0.200000 0.400000 0.200000 0.400000"
    assert (out / "train" / "labels" / "oi_jar.txt").read_text(encoding="utf-8") == ""
    assert (out / "train" / "images" / "oi_jar.jpg").is_symlink()
    assert not (out / "train" / "labels" / "oi_pos.txt").exists()
