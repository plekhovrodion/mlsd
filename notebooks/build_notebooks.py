"""Собирает ноутбуки ДЗ №3 для Kaggle из ячеек ниже.

Запуск: ``python notebooks/build_notebooks.py && ruff format notebooks``.
Ноутбуки сохраняются без выходов; выполненные версии загружаются из Kaggle после прогона.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/plekhovrodion/mlsd"

INSTALL = f'''import subprocess
import sys

# Ветка или коммит репозитория с пакетом photoquest_eval.
REPO_REF = "hw_3"
_ = subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        f"photoquest-eval[yolo] @ git+{REPO}@{{REPO_REF}}",
    ]
)
REPO_COMMIT = subprocess.check_output(
    ["git", "ls-remote", "{REPO}", REPO_REF], text=True
).split()[0]
print("photoquest_eval:", REPO_REF, REPO_COMMIT)'''

COMMON = """import hashlib
import importlib.metadata
import json
import platform
import shutil  # noqa: F401 — ячейка общая для обоих ноутбуков
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from photoquest_eval.config import (  # noqa: F401 — ячейка общая для обоих ноутбуков
    HOME_CLASSES,
    IMAGE_SIZES,
    SEED,
    TARGETS,
    YOLO_SETTINGS,
)

STARTED_UTC = datetime.now(UTC).isoformat()
STARTED = time.perf_counter()
np.random.seed(SEED)
torch.manual_seed(SEED)
if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Settings → Accelerator → GPU T4")
print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(*names):
    return {name: importlib.metadata.version(name) for name in names}"""

CHECKPOINTS = '''CHECKPOINTS = {
    "yolov8n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    ),
    "yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    ),
}


def get_checkpoint(filename: str, folder: Path) -> Path:
    """Скачивает предобученные веса и сверяет SHA-256 с прогоном ДЗ №2."""
    url, expected = CHECKPOINTS[filename]
    path = folder / filename
    folder.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urllib.request.urlretrieve(url, path)
    if sha256(path) != expected:
        raise ValueError(f"Не совпал SHA-256 весов {filename}")
    return path'''

TRAIN_CELLS = [
    (
        "markdown",
        """# ДЗ №3. Дообучение YOLO11n на домашних классах

Ноутбук готовит подвыборку COCO train2017 по 10 домашним классам, дообучает YOLO11n
и экспортирует веса в ONNX для трёх размеров входа. Оценка моделей вынесена
в `03_evaluate_models.ipynb`, чтобы обучение и оценка помещались в сессию Kaggle.

**Настройки Kaggle:** Accelerator — GPU T4, Internet — On.
**Входные данные:** датасет COCO 2017 с `train2017` и `instances_train2017.json`.""",
    ),
    ("code", INSTALL),
    ("code", COMMON),
    (
        "code",
        """from ultralytics import YOLO

from photoquest_eval.data import coco_to_yolo, write_yolo_yaml

OUTPUT = Path("/kaggle/working/hw3_train")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Гиперпараметры фиксируются до обучения и сохраняются в metadata.json.
TRAIN_SETTINGS = {
    "base_model": "yolo11n.pt",
    "epochs": 30,
    "patience": 10,
    "imgsz": 640,
    "batch": 32,
    "optimizer": "auto",
    "seed": SEED,
    "deterministic": True,
    "workers": 4,
    "save_period": 5,
}
DATA_SETTINGS = {"negative_fraction": 0.1, "val_fraction": 0.1, "seed": SEED}""",
    ),
    ("markdown", "## Данные\n\nИщем аннотации COCO train2017 во входных данных ноутбука."),
    (
        "code",
        """paths = sorted(Path("/kaggle/input").rglob("instances_train2017.json"))
if not paths:
    raise FileNotFoundError(
        "Добавьте во входные данные датасет COCO 2017 с train2017 "
        "(Add Input → Datasets → coco 2017)"
    )
TRAIN_ANNOTATIONS = paths[0]
coco_root = TRAIN_ANNOTATIONS.parent.parent
TRAIN_IMAGES = next(
    p for p in (coco_root / "train2017", coco_root / "images" / "train2017") if p.is_dir()
)
annotation_hash = sha256(TRAIN_ANNOTATIONS)
print(TRAIN_ANNOTATIONS, annotation_hash)
print(TRAIN_IMAGES, sum(1 for _ in TRAIN_IMAGES.glob("*.jpg")), "JPEG")""",
    ),
    (
        "code",
        """DATASET = Path("/kaggle/working/coco_home_yolo")
split_stats = coco_to_yolo(TRAIN_ANNOTATIONS, TRAIN_IMAGES, DATASET, TARGETS, **DATA_SETTINGS)
DATASET_YAML = write_yolo_yaml(DATASET, TARGETS)
display(pd.DataFrame(split_stats).T)
print(DATASET_YAML.read_text())""",
    ),
    ("markdown", "## Обучение"),
    ("code", CHECKPOINTS),
    (
        "code",
        """base = get_checkpoint(TRAIN_SETTINGS["base_model"], OUTPUT / "pretrained")
model = YOLO(str(base))
train_started = time.perf_counter()
results = model.train(
    data=str(DATASET_YAML),
    project=str(OUTPUT),
    name="yolo11n_home",
    exist_ok=True,
    plots=True,
    **{k: v for k, v in TRAIN_SETTINGS.items() if k != "base_model"},
)
train_seconds = time.perf_counter() - train_started
RUN = OUTPUT / "yolo11n_home"
BEST = RUN / "weights" / "best.pt"
print(f"Обучение заняло {train_seconds / 3600:.2f} ч, лучшие веса: {BEST}")""",
    ),
    (
        "markdown",
        """## Кривые обучения

Функция потерь Ultralytics для детекции складывается из трёх частей: `box` — ошибка
положения рамки (CIoU), `cls` — ошибка класса (бинарная кросс-энтропия),
`dfl` — distribution focal loss для границ рамки.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt

history = pd.read_csv(RUN / "results.csv")
history.columns = history.columns.str.strip()
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for axis, part in zip(axes[:3], ("box", "cls", "dfl"), strict=True):
    axis.plot(history["epoch"], history[f"train/{part}_loss"], label="train")
    axis.plot(history["epoch"], history[f"val/{part}_loss"], label="val")
    axis.set(title=f"{part}_loss", xlabel="эпоха")
    axis.legend()
axes[3].plot(history["epoch"], history["metrics/mAP50(B)"], label="mAP@0.50")
axes[3].plot(history["epoch"], history["metrics/mAP50-95(B)"], label="mAP@0.50:0.95")
axes[3].set(title="Качество на val", xlabel="эпоха")
axes[3].legend()
fig.tight_layout()
fig.savefig(OUTPUT / "training_curves.png", dpi=150)
plt.show()
history.to_csv(OUTPUT / "training_history.csv", index=False)
display(history.tail(3))""",
    ),
    (
        "markdown",
        "## Экспорт в ONNX\n\nДля браузера в ДЗ №4 готовим файлы для трёх размеров входа.",
    ),
    (
        "code",
        """EXPORTS = OUTPUT / "export"
EXPORTS.mkdir(exist_ok=True)
shutil.copy(BEST, EXPORTS / "yolo11n_home.pt")
export_rows = []
for size in IMAGE_SIZES:
    exported = Path(YOLO(str(BEST)).export(format="onnx", imgsz=size, simplify=True))
    target = EXPORTS / f"yolo11n_home_{size}.onnx"
    shutil.move(exported, target)
    export_rows.append(
        {"file": target.name, "imgsz": size, "MiB": target.stat().st_size / 2**20,
         "sha256": sha256(target)}
    )
export_rows.append(
    {"file": "yolo11n_home.pt", "imgsz": 640,
     "MiB": (EXPORTS / "yolo11n_home.pt").stat().st_size / 2**20,
     "sha256": sha256(EXPORTS / "yolo11n_home.pt")}
)
exports = pd.DataFrame(export_rows)
exports.to_csv(OUTPUT / "exports.csv", index=False)
display(exports)""",
    ),
    (
        "code",
        """metadata = {
    "repo_ref": REPO_REF,
    "repo_commit": REPO_COMMIT,
    "train_annotations_sha256": annotation_hash,
    "targets": list(TARGETS),
    "openimages_names": HOME_CLASSES,
    "data_settings": DATA_SETTINGS,
    "split_stats": split_stats,
    "train_settings": TRAIN_SETTINGS,
    "best_epoch": int(history.loc[history["metrics/mAP50-95(B)"].idxmax(), "epoch"]),
    "epochs_run": int(history["epoch"].max()),
    "train_seconds": round(train_seconds, 1),
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "packages": package_versions(
        "torch", "ultralytics", "numpy", "onnx", "onnxruntime", "onnxslim"
    ),
    "started_utc": STARTED_UTC,
    "elapsed_seconds": round(time.perf_counter() - STARTED, 1),
}
(OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
# В архив для репозитория идут метрики, графики и экспорт; промежуточные чекпоинты — нет.
with zipfile.ZipFile("/kaggle/working/hw3_train_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob("*")):
        relative = path.relative_to(OUTPUT)
        skipped = {"pretrained", "weights"} & set(relative.parts)
        if path.is_file() and not skipped:
            z.write(path, relative)
print(json.dumps(metadata, indent=2, ensure_ascii=False))""",
    ),
]

EVAL_CELLS = [
    (
        "markdown",
        """# ДЗ №3. Оценка baseline и основной модели

Сравниваем три модели по одному протоколу:

- **YOLOv8n** — предобученный baseline;
- **YOLO11n** — предобученный, показывает вклад архитектуры;
- **YOLO11n-home** — дообученный на 10 домашних классах (`02_train_yolo11n.ipynb`).

Каждая модель проверяется на размерах входа 640, 416 и 320 и на двух наборах:
COCO val2017 (протокол ДЗ №2, пороги по фолдам) и Open Images V7 test
(перенос на другой источник; пороги взяты с COCO и здесь не подбираются).

**Настройки Kaggle:** GPU T4, Internet — On.
**Входные данные:** COCO 2017 val (как в ДЗ №2) и вывод ноутбука обучения.""",
    ),
    ("code", INSTALL),
    ("code", COMMON),
    (
        "code",
        """import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image

from photoquest_eval.data import (
    OPENIMAGES_URLS,
    download_openimages,
    load_coco,
    load_openimages,
    sample_openimages,
)
from photoquest_eval.evaluate import coco_oof, fit_thresholds, transfer_metrics
from photoquest_eval.inference import predict_scores
from photoquest_eval.metrics import paired_bootstrap_diff

OUTPUT = Path("/kaggle/working/hw3_eval")
OUTPUT.mkdir(parents=True, exist_ok=True)
RULE_VERSION = "hw3-r1"
OI_SAMPLE = {"positives_per_class": 400, "negatives_per_class": 800, "seed": SEED}""",
    ),
    ("markdown", "## Данные\n\n### COCO val2017"),
    (
        "code",
        """import kagglehub

COCO_ROOT = Path(kagglehub.dataset_download("nikitamanaenkov/microsoft-coco-2017"))
VAL_ANNOTATIONS = sorted(COCO_ROOT.rglob("instances_val2017.json"))[0]
assert sha256(VAL_ANNOTATIONS) == (
    "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
), "Снимок аннотаций отличается от ДЗ №2"
val_root = VAL_ANNOTATIONS.parent.parent
VAL_IMAGES = next(
    p for p in (val_root / "val2017", val_root / "images" / "val2017") if p.is_dir()
)
coco = load_coco(VAL_ANNOTATIONS, VAL_IMAGES, TARGETS)
print(len(coco.image_ids), "фото;", "положительных по классам:",
      dict(zip(TARGETS, coco.labels.sum(axis=0).tolist(), strict=True)))""",
    ),
    (
        "markdown",
        """### Open Images V7 test

Выборка фиксируется до запуска моделей: для каждого класса до 400 фото с проверенной
положительной меткой и до 800 с проверенной отрицательной. Фото без проверенной
метки класса в его оценке не участвуют. При скачивании фото уменьшаются до 1024 px
по длинной стороне, чтобы уложиться в лимит диска Kaggle.""",
    ),
    (
        "code",
        """OI_DIR = Path("/kaggle/working/openimages")
OI_DIR.mkdir(exist_ok=True)
csv_paths = {}
for key, name in (
    ("image_labels", "test-human-imagelabels.csv"),
    ("boxes", "test-annotations-bbox.csv"),
):
    csv_paths[key] = OI_DIR / name
    if not csv_paths[key].exists():
        urllib.request.urlretrieve(OPENIMAGES_URLS[key], csv_paths[key])
META_URL = "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv"
csv_paths["meta"] = OI_DIR / "test-images-with-rotation.csv"
if not csv_paths["meta"].exists():
    urllib.request.urlretrieve(META_URL, csv_paths["meta"])
{key: sha256(path) for key, path in csv_paths.items()}""",
    ),
    (
        "code",
        """oi_ids = sample_openimages(csv_paths["image_labels"], TARGETS, **OI_SAMPLE)
failed = download_openimages(oi_ids, OI_DIR / "images")
print(f"Выбрано {len(oi_ids):,} фото, не скачалось {len(failed)}")
oi = load_openimages(csv_paths["image_labels"], csv_paths["boxes"], OI_DIR / "images", TARGETS)
oi_meta = pd.read_csv(csv_paths["meta"]).set_index("ImageID")
manifest = oi_meta.reindex(oi.image_ids)[
    ["License", "Author", "OriginalLandingURL", "Rotation"]
].reset_index(names="ImageID")
manifest.to_csv(OUTPUT / "openimages_manifest.csv", index=False)
known = (oi.labels >= 0)
display(
    pd.DataFrame(
        {
            "positive_images": (oi.labels == 1).sum(axis=0),
            "verified_negatives": (oi.labels == 0).sum(axis=0),
            "unknown": (~known).sum(axis=0),
        },
        index=TARGETS,
    )
)
print("Фото с ненулевым Rotation:", int(manifest["Rotation"].fillna(0).ne(0).sum()))""",
    ),
    ("markdown", "## Модели и score"),
    ("code", CHECKPOINTS),
    (
        "code",
        """FT_PATHS = sorted(Path("/kaggle/input").rglob("yolo11n_home.pt"))
if not FT_PATHS:
    raise FileNotFoundError("Добавьте во входные данные вывод ноутбука 02_train_yolo11n")
FT_DIR = FT_PATHS[0].parent
MODELS = {
    "yolov8n": get_checkpoint("yolov8n.pt", OUTPUT / "pretrained"),
    "yolo11n": get_checkpoint("yolo11n.pt", OUTPUT / "pretrained"),
    "yolo11n_home": FT_PATHS[0],
}
model_hashes = {name: sha256(path) for name, path in MODELS.items()}
model_hashes""",
    ),
    (
        "code",
        """scores = {}
timings = []
for name, weights in MODELS.items():
    for size in IMAGE_SIZES:
        for dataset in (coco, oi):
            started = time.perf_counter()
            scores[(dataset.name, name, size)] = predict_scores(
                weights, dataset.image_paths, TARGETS, size, device=0, **YOLO_SETTINGS
            )
            seconds = time.perf_counter() - started
            timings.append(
                {"dataset": dataset.name, "model": name, "image_size": size,
                 "images": len(dataset.image_ids), "seconds": seconds}
            )
            print(f"{dataset.name:10} {name:13} {size}: {seconds:.0f} с")
np.savez_compressed(
    OUTPUT / "scores.npz",
    **{"__".join(map(str, key)): value for key, value in scores.items()},
)
pd.DataFrame(timings).to_csv(OUTPUT / "timings_t4.csv", index=False)""",
    ),
    (
        "markdown",
        """## COCO val2017: протокол ДЗ №2

Пороги подбираются по отрицательным примерам четырёх фолдов и проверяются на пятом.
Интервалы — перцентильный bootstrap по фото (1 000 повторов).""",
    ),
    (
        "code",
        """coco_rows, calibration_rows = [], []
for name in MODELS:
    for size in IMAGE_SIZES:
        rows, calibration, oof = coco_oof(coco, scores[("coco", name, size)], name, size)
        coco_rows.extend(rows)
        calibration_rows.extend(calibration)
        np.savez_compressed(OUTPUT / f"coco_oof_{name}_{size}.npz", **oof)
coco_metrics = pd.DataFrame(coco_rows)
coco_metrics.to_csv(OUTPUT / "coco_per_class.csv", index=False)
pd.DataFrame(calibration_rows).to_csv(OUTPUT / "coco_calibration_folds.csv", index=False)


def macro(table):
    return (
        table.groupby(["dataset", "model", "image_size", "protocol"], sort=False)
        .agg(
            macro_ap=("average_precision", "mean"),
            macro_auc=("roc_auc", "mean"),
            recall=("recall", "mean"),
            far=("far", "mean"),
            classes_far_above_5pct=("far_exceeds_target", "sum"),
            recall_main=("recall_main", "mean"),
            recall_background=("recall_background", "mean"),
            recall_small_only=("recall_small_only", "mean"),
        )
        .reset_index()
    )


coco_summary = macro(coco_metrics)
display(coco_summary.round(4))""",
    ),
    (
        "markdown",
        """## Пороги для переноса и для игры

Итоговые пороги считаются по всем отрицательным фото COCO val2017 с консервативным
протоколом `confidence_95pct`. Для YOLO11n-home они сохраняются в `thresholds_<size>.json`:
этот файл в ДЗ №4 загрузит браузер.""",
    ),
    (
        "code",
        """thresholds = {}
for name in MODELS:
    for size in IMAGE_SIZES:
        thresholds[(name, size)] = fit_thresholds(
            coco, scores[("coco", name, size)], f"{name}-{model_hashes[name][:12]}",
            size, RULE_VERSION,
        )
for size in IMAGE_SIZES:
    thresholds[("yolo11n_home", size)].to_json(OUTPUT / f"thresholds_{size}.json")
print((OUTPUT / "thresholds_640.json").read_text())""",
    ),
    (
        "markdown",
        """## Open Images V7 test: перенос

Негативы — только фото с проверенной отрицательной меткой. Срезы положительных фото:
`main` — рамка не меньше 5% кадра и в центральной зоне, `background` — остальные,
`small_only` — все рамки класса меньше 1% кадра.""",
    ),
    (
        "code",
        """oi_rows, oi_predictions = [], {}
for name in MODELS:
    for size in IMAGE_SIZES:
        rows, predicted = transfer_metrics(
            oi, scores[("openimages", name, size)], thresholds[(name, size)], name
        )
        oi_rows.extend(rows)
        oi_predictions[(name, size)] = predicted
oi_metrics = pd.DataFrame(oi_rows)
oi_metrics.to_csv(OUTPUT / "openimages_per_class.csv", index=False)
oi_summary = macro(oi_metrics)
summary = pd.concat([coco_summary, oi_summary], ignore_index=True)
summary.to_csv(OUTPUT / "model_metrics.csv", index=False)
display(oi_summary.round(4))""",
    ),
    (
        "markdown",
        """## Значимость различий

Парный bootstrap на одних и тех же фото Open Images (размер входа 640):
разница recall и FAR моделей по каждому классу с 95% интервалом.""",
    ),
    (
        "code",
        """pairs = [
    ("yolo11n_home", "yolov8n"),
    ("yolo11n_home", "yolo11n"),
    ("yolo11n", "yolov8n"),
]
diff_rows = []
for a, b in pairs:
    for column, target in enumerate(TARGETS):
        known = oi.mask(column)
        y = oi.labels[known, column] == 1
        diff = paired_bootstrap_diff(
            y, oi_predictions[(a, 640)][target][known], oi_predictions[(b, 640)][target][known]
        )
        for metric in ("recall", "far"):
            estimate, low, high = diff[metric]
            diff_rows.append(
                {"model_a": a, "model_b": b, "target": target, "metric": metric,
                 "diff": estimate, "low": low, "high": high,
                 "significant": not (low <= 0 <= high)}
            )
significance = pd.DataFrame(diff_rows)
significance.to_csv(OUTPUT / "openimages_paired_bootstrap.csv", index=False)
display(
    significance.groupby(["model_a", "model_b", "metric"])
    .agg(mean_diff=("diff", "mean"), significant_classes=("significant", "sum"))
    .round(4)
)""",
    ),
    ("markdown", "## Графики"),
    (
        "code",
        """conservative = oi_metrics[oi_metrics["protocol"].eq("confidence_95pct")]
part = conservative[conservative["image_size"].eq(640)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for axis, metric, title, upper, cmap in (
    (axes[0], "recall", "Recall на Open Images", 1.0, "Blues"),
    (axes[1], "far", "FAR на Open Images", 0.1, "OrRd"),
):
    table = part.pivot(index="target", columns="model", values=metric).reindex(TARGETS)
    image = axis.imshow(table, vmin=0, vmax=upper, aspect="auto", cmap=cmap)
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=20)
    axis.set_yticks(range(len(TARGETS)), table.index)
    axis.set_title(title)
    for r in range(len(TARGETS)):
        for c in range(len(table.columns)):
            value = table.iloc[r, c]
            axis.text(c, r, f"{value:.1%}", ha="center", va="center", fontsize=8,
                      color="white" if value > upper * 0.6 else "black")
    fig.colorbar(image, ax=axis, shrink=0.8, format=PercentFormatter(1))
fig.suptitle("Перенос на Open Images, вход 640, консервативный порог")
fig.tight_layout()
fig.savefig(OUTPUT / "openimages_recall_far.png", dpi=150)
plt.show()""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cons = summary[summary["protocol"].eq("confidence_95pct")]
for axis, dataset in zip(axes, ("coco", "openimages"), strict=True):
    for name in MODELS:
        rows = cons[cons["dataset"].eq(dataset) & cons["model"].eq(name)]
        axis.plot(rows["image_size"], rows["recall"], marker="o", label=name)
    axis.set(title=f"Recall от размера входа — {dataset}", xlabel="imgsz", ylabel="recall")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "recall_by_image_size.png", dpi=150)
plt.show()

slices = oi_summary[oi_summary["image_size"].eq(640)].set_index("model")[
    ["recall_main", "recall_background", "recall_small_only"]
]
ax = slices.plot.bar(figsize=(8, 4), rot=0, title="Recall по срезам, Open Images, 640")
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.savefig(OUTPUT / "openimages_slices.png", dpi=150)
plt.show()""",
    ),
    (
        "markdown",
        """## Разбор ошибок YOLO11n-home

Для рюкзака, телефона и бутылки берём на Open Images (640) ложное подтверждение
с наибольшим score и пропуск с наименьшим. Авторы и лицензии показанных фото —
в `error_cases.csv`.""",
    ),
    (
        "code",
        """ERROR_TARGETS = ["backpack", "cell phone", "bottle"]
oi_scores = scores[("openimages", "yolo11n_home", 640)]
records = []
for target in ERROR_TARGETS:
    column = TARGETS.index(target)
    known = oi.mask(column)
    truth = oi.labels[:, column] == 1
    predicted = oi_predictions[("yolo11n_home", 640)][target]
    for kind, mask in (
        ("false_positive", predicted & ~truth & known),
        ("false_negative", ~predicted & truth & known),
    ):
        index = np.flatnonzero(mask)
        for row in index[np.argsort(oi_scores[index, column], kind="stable")]:
            records.append({"target": target, "error_type": kind,
                            "image_id": oi.image_ids[row], "score": float(oi_scores[row, column]),
                            "cutoff": thresholds[("yolo11n_home", 640)].rules[target].high})
error_cases = pd.DataFrame(records).merge(manifest, left_on="image_id", right_on="ImageID")
error_cases.to_csv(OUTPUT / "error_cases.csv", index=False)
examples = pd.concat(
    [
        error_cases[error_cases["error_type"].eq("false_positive")]
        .sort_values("score", ascending=False).groupby("target").head(1),
        error_cases[error_cases["error_type"].eq("false_negative")]
        .sort_values("score").groupby("target").head(1),
    ]
).sort_values(["target", "error_type"])
fig, axes = plt.subplots(len(ERROR_TARGETS), 2, figsize=(12, 11))
for axis, row in zip(axes.flat, examples.itertuples(), strict=True):
    with Image.open(OI_DIR / "images" / f"{row.image_id}.jpg") as image:
        axis.imshow(image.convert("RGB"))
    kind = "Ложное подтверждение" if row.error_type == "false_positive" else "Пропуск"
    axis.set_title(f"{row.target} — {kind}\\n{row.image_id}, score {row.score:.3f}, "
                   f"порог {row.cutoff:.3f}\\n© {row.Author}, CC BY 2.0", fontsize=9)
    axis.axis("off")
fig.tight_layout()
fig.savefig(OUTPUT / "error_examples.jpg", dpi=110, pil_kwargs={"quality": 85})
plt.show()
display(error_cases.groupby(["target", "error_type"]).size().unstack())""",
    ),
    (
        "markdown",
        """## Совпадение ONNX и PyTorch

Для ДЗ №4 модель будет работать в браузере из ONNX. Сравниваем score и решения
ONNX-версий YOLO11n-home с PyTorch на 300 фото COCO val2017.""",
    ),
    (
        "code",
        """rng = np.random.default_rng(SEED)
sample = sorted(rng.choice(len(coco.image_paths), 300, replace=False).tolist())
sample_paths = [coco.image_paths[i] for i in sample]
parity_rows = []
for size in IMAGE_SIZES:
    onnx_path = FT_DIR / f"yolo11n_home_{size}.onnx"
    onnx_scores = predict_scores(onnx_path, sample_paths, TARGETS, size, batch_size=1,
                                 device="cpu", **YOLO_SETTINGS)
    torch_scores = scores[("coco", "yolo11n_home", size)][sample]
    rule = thresholds[("yolo11n_home", size)]
    cutoffs = np.array([rule.rules[t].high for t in TARGETS])
    parity_rows.append(
        {"image_size": size, "max_abs_score_diff": float(np.abs(onnx_scores - torch_scores).max()),
         "decision_agreement": float(((onnx_scores > cutoffs) == (torch_scores > cutoffs)).mean()),
         "onnx_sha256": sha256(onnx_path)}
    )
parity = pd.DataFrame(parity_rows)
parity.to_csv(OUTPUT / "onnx_parity.csv", index=False)
display(parity)""",
    ),
    (
        "code",
        """metadata = {
    "repo_ref": REPO_REF,
    "repo_commit": REPO_COMMIT,
    "targets": list(TARGETS),
    "rule_version": RULE_VERSION,
    "yolo_settings": YOLO_SETTINGS,
    "image_sizes": list(IMAGE_SIZES),
    "openimages_sample": OI_SAMPLE,
    "openimages_images": len(oi.image_ids),
    "openimages_failed_downloads": failed,
    "openimages_csv_sha256": {key: sha256(path) for key, path in csv_paths.items()},
    "coco_val_images": len(coco.image_ids),
    "model_sha256": model_hashes,
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "packages": package_versions(
        "torch", "ultralytics", "numpy", "pandas", "scikit-learn", "scipy", "onnxruntime"
    ),
    "started_utc": STARTED_UTC,
    "elapsed_seconds": round(time.perf_counter() - STARTED, 1),
}
(OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
with zipfile.ZipFile("/kaggle/working/hw3_eval_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and "pretrained" not in path.relative_to(OUTPUT).parts:
            z.write(path, path.relative_to(OUTPUT))
print(json.dumps(metadata, indent=2, ensure_ascii=False))""",
    ),
]


def build(cells, path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.strip("\n").splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, source in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build(TRAIN_CELLS, HERE / "02_train_yolo11n.ipynb")
    build(EVAL_CELLS, HERE / "03_evaluate_models.ipynb")
