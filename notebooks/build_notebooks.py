"""Собирает ноутбуки ДЗ №3 для Kaggle из ячеек ниже.

Запуск: ``python notebooks/build_notebooks.py && ruff format notebooks``.
Ноутбуки сохраняются без выходов; выполненные версии загружаются из Kaggle после прогона.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/plekhovrodion/mlsd"

INSTALL = f"""import subprocess
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
print("photoquest_eval:", REPO_REF, REPO_COMMIT)"""

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
    "yolo11s.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
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

CURVES = """import matplotlib.pyplot as plt

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
display(history.tail(3))"""

EXPORT = """EXPORTS = OUTPUT / "export"
EXPORTS.mkdir(exist_ok=True)
shutil.copy(BEST, EXPORTS / f"{RUN_NAME}.pt")
export_rows = []
for size in IMAGE_SIZES:
    exported = Path(YOLO(str(BEST)).export(format="onnx", imgsz=size, simplify=True))
    target = EXPORTS / f"{RUN_NAME}_{size}.onnx"
    shutil.move(exported, target)
    export_rows.append(
        {"file": target.name, "imgsz": size, "MiB": target.stat().st_size / 2**20,
         "sha256": sha256(target)}
    )
export_rows.append(
    {"file": f"{RUN_NAME}.pt", "imgsz": 640,
     "MiB": (EXPORTS / f"{RUN_NAME}.pt").stat().st_size / 2**20,
     "sha256": sha256(EXPORTS / f"{RUN_NAME}.pt")}
)
exports = pd.DataFrame(export_rows)
exports.to_csv(OUTPUT / "exports.csv", index=False)
display(exports)"""

TRADEOFF = """from sklearn.metrics import roc_auc_score

from photoquest_eval.calibration import threshold_for_far
from photoquest_eval.tradeoff import mean_rates, two_fold_tradeoff

HN_LEVELS = (0.40, 0.20, 0.10)
HN_FOLD = np.random.default_rng(SEED).random(len(oi.image_ids)) < 0.5


def coco_floor(model_scores, column):
    negatives = model_scores[coco.labels[:, column] == 0, column]
    return threshold_for_far(negatives, "confidence_95pct", 0.05, 0.95)


def tradeoff_rows(name, oi_scores, coco_scores, floors):
    rows = []
    for column, target in enumerate(TARGETS):
        labels = oi.labels[:, column]
        known = labels >= 0
        row = {
            "model": name,
            "target": target,
            "oi_auc_hardneg": roc_auc_score(labels[known], oi_scores[known, column]),
        }
        for point in two_fold_tradeoff(labels, oi_scores[:, column], floors[column], HN_LEVELS, HN_FOLD):
            tag = int(point.level * 100)
            coco_recall, coco_far = mean_rates(coco.labels[:, column], coco_scores[:, column], point.cutoffs)
            row |= {
                f"oi_recall@hn{tag}": point.recall,
                f"oi_far@hn{tag}": point.far,
                f"coco_recall@hn{tag}": coco_recall,
                f"coco_far@hn{tag}": coco_far,
            }
        rows.append(row)
    return rows


hn_rows = []
for name in MODELS:
    coco_scores = scores[("coco", name, 640)]
    floors = [coco_floor(coco_scores, column) for column in range(len(TARGETS))]
    hn_rows += tradeoff_rows(name, scores[("openimages", name, 640)], coco_scores, floors)
hardneg_tradeoff = pd.DataFrame(hn_rows)
hardneg_tradeoff.to_csv(OUTPUT / "hardneg_tradeoff_per_class.csv", index=False)
display(hardneg_tradeoff.groupby("model", sort=False).mean(numeric_only=True).round(3).T)
display(hardneg_tradeoff.pivot(index="target", columns="model", values="oi_auc_hardneg").round(3))"""

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
RUN_NAME = "yolo11n_home"

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
# Фото копируются на локальный диск и уменьшаются до размера входа: чтение из
# /kaggle/input идёт с сетевого хранилища и замедляет каждую эпоху в разы.
DATA_SETTINGS = {
    "negative_fraction": 0.1,
    "val_fraction": 0.1,
    "seed": SEED,
    "max_side": TRAIN_SETTINGS["imgsz"],
}""",
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
        """# /tmp — локальный диск сессии; он не попадает в вывод ноутбука и не занимает
# лимит /kaggle/working. Повторный запуск пропускает уже скопированные фото.
DATASET = Path("/tmp/coco_home_yolo")
copy_started = time.perf_counter()
split_stats = coco_to_yolo(TRAIN_ANNOTATIONS, TRAIN_IMAGES, DATASET, TARGETS, **DATA_SETTINGS)
copy_seconds = time.perf_counter() - copy_started
print(f"Подготовка выборки заняла {copy_seconds / 60:.1f} мин")
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
    name=RUN_NAME,
    exist_ok=True,
    plots=True,
    **{k: v for k, v in TRAIN_SETTINGS.items() if k != "base_model"},
)
train_seconds = time.perf_counter() - train_started
RUN = OUTPUT / RUN_NAME
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
    ("code", CURVES),
    (
        "markdown",
        "## Экспорт в ONNX\n\nДля браузера в ДЗ №4 готовим файлы для трёх размеров входа.",
    ),
    ("code", EXPORT),
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
    "data_prep_seconds": round(copy_seconds, 1),
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

_HARDNEG_TEMPLATE = [
    (
        "markdown",
        """# ДЗ №3. YOLO11n на трудных негативах Open Images

Оценка `03_evaluate_models.ipynb` показала, что порог, подобранный на COCO, почти не защищает
от похожих предметов: на проверенных негативах Open Images (банка вместо бутылки) FAR
около 62% у всех моделей. Первое дообучение на COCO (`02_train_yolo11n.ipynb`) этого
не исправило: те же фото и метки, на которых YOLO11n уже обучен.

Здесь к подвыборке COCO train добавляются фото Open Images **train** (test остаётся
для оценки):

- **трудные негативы** — проверенная человеком отрицательная метка класса; такие фото
  попали в разметку как кандидаты классификатора, то есть на них похожий предмет;
- **положительные** — проверенная положительная метка с рамками, чтобы модель
  не выучила правило «стиль Open Images — предметов нет».

Фото, где у проверенно положительного класса нет рамок, пропускаются. Непроверенные
классы на фото Open Images остаются риском шума в разметке.

Чтобы не повторить провал первых эпох прошлого обучения (mAP50 на val упал с 0,33 до 0,18),
оптимизатор задан явно: SGD с lr0 = 0,002, разогрев 1 эпоха, первые 10 слоёв заморожены.

**Критерий успеха (зафиксирован до запуска).** На Open Images test FAR при пороге с COCO
снижается с 62% хотя бы до 40% (в среднем по классам, вход 640), а recall на COCO val2017
при консервативном пороге теряет не больше 2 п.п. относительно предобученного YOLO11n (83,1%).

**Настройки Kaggle:** Accelerator — GPU T4, Internet — On.
**Входные данные:** датасет COCO 2017 с `train2017` и `instances_train2017.json`.""",
    ),
    ("code", INSTALL),
    ("code", COMMON),
    (
        "code",
        """from ultralytics import YOLO

from photoquest_eval.data import (
    OPENIMAGES_URLS,
    boxable_image_ids,
    coco_to_yolo,
    download_openimages,
    openimages_to_yolo,
    sample_hard_negatives,
    scan_openimages_boxes,
    scan_openimages_labels,
    write_yolo_yaml,
)

OUTPUT = Path("/kaggle/working/hw3_train_hardneg")
OUTPUT.mkdir(parents=True, exist_ok=True)
RUN_NAME = "yolo11n_hardneg"

# Гиперпараметры фиксируются до обучения и сохраняются в metadata.json.
TRAIN_SETTINGS = {
    "base_model": "yolo11n.pt",
    "epochs": 30,
    "patience": 10,
    "imgsz": 640,
    "batch": 32,
    "optimizer": "SGD",
    "lr0": 0.002,
    "lrf": 0.1,
    "warmup_epochs": 1,
    "freeze": 10,
    "seed": SEED,
    "deterministic": True,
    "workers": 4,
    "save_period": 5,
}
DATA_SETTINGS = {
    "negative_fraction": 0.1,
    "val_fraction": 0.1,
    "seed": SEED,
    "max_side": TRAIN_SETTINGS["imgsz"],
}
OI_TRAIN_SAMPLE = {"positives_per_class": 800, "negatives_per_class": 1500, "seed": SEED}""",
    ),
    ("markdown", "## Данные\n\n### COCO train2017 — как в `02_train_yolo11n.ipynb`"),
    (
        "code",
        """paths = sorted(Path("/kaggle/input").rglob("instances_train2017.json"))
if not paths:
    raise FileNotFoundError("Добавьте во входные данные датасет COCO 2017 с train2017")
TRAIN_ANNOTATIONS = paths[0]
coco_root = TRAIN_ANNOTATIONS.parent.parent
TRAIN_IMAGES = next(
    p for p in (coco_root / "train2017", coco_root / "images" / "train2017") if p.is_dir()
)
annotation_hash = sha256(TRAIN_ANNOTATIONS)
DATASET = Path("/tmp/hardneg_yolo")
copy_started = time.perf_counter()
coco_stats = coco_to_yolo(TRAIN_ANNOTATIONS, TRAIN_IMAGES, DATASET, TARGETS, **DATA_SETTINGS)
print(f"COCO: {(time.perf_counter() - copy_started) / 60:.1f} мин")
display(pd.DataFrame(coco_stats).T)""",
    ),
    (
        "markdown",
        """### Open Images train

Файлы разметки train весят 2,7 и 2,3 ГБ; из них читаются только строки 10 целевых классов.""",
    ),
    (
        "code",
        """OI_CACHE = Path("/tmp/openimages_train")
OI_CACHE.mkdir(exist_ok=True)
oi_started = time.perf_counter()
csv_paths = {}
for key in ("train_image_labels", "train_boxes"):
    csv_paths[key] = OI_CACHE / Path(OPENIMAGES_URLS[key]).name
    if not csv_paths[key].exists():
        urllib.request.urlretrieve(OPENIMAGES_URLS[key], csv_paths[key])
oi_status = scan_openimages_labels(csv_paths["train_image_labels"], TARGETS)
# На S3 лежат только фото с рамками (~1,7 млн из ~9 млн с метками): выбираем только из них.
boxable = boxable_image_ids(csv_paths["train_boxes"])
labelled_images = len(oi_status)
oi_status = {image_id: s for image_id, s in oi_status.items() if image_id in boxable}
print(f"Фото с метками целевых классов: {labelled_images:,}, из них с рамками: {len(oi_status):,}")
availability = pd.DataFrame(
    {
        "verified_positive": [sum(s.get(t) == 1 for s in oi_status.values()) for t in TARGETS],
        "verified_negative": [sum(s.get(t) == 0 for s in oi_status.values()) for t in TARGETS],
    },
    index=TARGETS,
)
display(availability)
oi_ids = sample_hard_negatives(oi_status, TARGETS, **OI_TRAIN_SAMPLE)
oi_boxes = scan_openimages_boxes(csv_paths["train_boxes"], oi_ids, TARGETS)
failed = download_openimages(
    oi_ids, OI_CACHE / "images", max_side=TRAIN_SETTINGS["imgsz"], split="train"
)
oi_stats = openimages_to_yolo(
    oi_ids, oi_status, oi_boxes, OI_CACHE / "images", DATASET, TARGETS,
    DATA_SETTINGS["val_fraction"], SEED,
)
oi_seconds = time.perf_counter() - oi_started
print(f"Open Images: выбрано {len(oi_ids):,}, не скачалось {len(failed)}, "
      f"{oi_seconds / 60:.1f} мин")
display(pd.DataFrame(oi_stats).T)
pd.Series(oi_ids, name="ImageID").to_csv(OUTPUT / "openimages_train_ids.csv", index=False)
DATASET_YAML = write_yolo_yaml(DATASET, TARGETS)""",
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
    name=RUN_NAME,
    exist_ok=True,
    plots=True,
    **{k: v for k, v in TRAIN_SETTINGS.items() if k != "base_model"},
)
train_seconds = time.perf_counter() - train_started
RUN = OUTPUT / RUN_NAME
BEST = RUN / "weights" / "best.pt"
print(f"Обучение заняло {train_seconds / 3600:.2f} ч, лучшие веса: {BEST}")""",
    ),
    ("markdown", "## Кривые обучения"),
    ("code", CURVES),
    ("markdown", "## Экспорт в ONNX"),
    ("code", EXPORT),
    (
        "code",
        """metadata = {
    "repo_ref": REPO_REF,
    "repo_commit": REPO_COMMIT,
    "train_annotations_sha256": annotation_hash,
    "targets": list(TARGETS),
    "data_settings": DATA_SETTINGS,
    "coco_split_stats": coco_stats,
    "openimages_train_sample": OI_TRAIN_SAMPLE,
    "openimages_train_images": len(oi_ids),
    "openimages_boxable_only": True,
    "openimages_labelled_images": labelled_images,
    "openimages_boxable_labelled_images": len(oi_status),
    "openimages_failed_downloads": len(failed),
    "openimages_split_stats": oi_stats,
    "openimages_csv_sha256": {key: sha256(path) for key, path in csv_paths.items()},
    "train_settings": TRAIN_SETTINGS,
    "best_epoch": int(history.loc[history["metrics/mAP50-95(B)"].idxmax(), "epoch"]),
    "epochs_run": int(history["epoch"].max()),
    "openimages_prep_seconds": round(oi_seconds, 1),
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
archive = Path(f"/kaggle/working/{OUTPUT.name}_results.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob("*")):
        relative = path.relative_to(OUTPUT)
        if path.is_file() and not {"pretrained", "weights"} & set(relative.parts):
            z.write(path, relative)
print(json.dumps(metadata, indent=2, ensure_ascii=False))""",
    ),
]


def hardneg_cells(intro: str, output: str, run_name: str, base_model: str) -> list:
    """Ноутбук обучения на трудных негативах с заданными моделью и именем запуска."""
    swaps = [
        (
            'OUTPUT = Path("/kaggle/working/hw3_train_hardneg")',
            f'OUTPUT = Path("/kaggle/working/{output}")',
        ),
        ('RUN_NAME = "yolo11n_hardneg"', f'RUN_NAME = "{run_name}"'),
        ('"base_model": "yolo11n.pt",', f'"base_model": "{base_model}",'),
    ]
    cells = [("markdown", intro)]
    for kind, source in _HARDNEG_TEMPLATE[1:]:
        for old, new in swaps:
            source = source.replace(old, new)
        cells.append((kind, source))
    return cells


HARDNEG_COMMON_MD = """
Данные как в первом запуске (`yolo11n_hardneg`, коммит `41e6245`), с одной правкой: фото
Open Images train выбираются только из подмножества с рамками. Человеческие метки уровня
фото есть примерно для 9 млн снимков, а на S3 лежат только ~1,7 млн фото с рамками; в первом
запуске из-за этого не скачались 56% выбранных фото и в обучение попало ~7,7 тыс. вместо ~20 тыс.

- **трудные негативы** — проверенная человеком отрицательная метка класса: фото попало
  в разметку как кандидат классификатора, то есть на нём похожий предмет;
- **положительные** — проверенная положительная метка с рамками.

Оптимизатор и расписание как в первом запуске: SGD, lr0 = 0,002, разогрев 1 эпоха,
первые 10 слоёв заморожены, 30 эпох.

**Настройки Kaggle:** Accelerator — GPU T4, Internet — On.
**Входные данные:** датасет COCO 2017 с `train2017` и `instances_train2017.json`."""

HARDNEG_V2_CELLS = hardneg_cells(
    """# ДЗ №3. YOLO11n на трудных негативах Open Images, v2

Первый запуск (`yolo11n_hardneg`) улучшил разделение предмета и похожего предмета
(AUC на трудных негативах Open Images 0,72 → 0,75) и recall, но FAR на похожих предметах
при пороге COCO не снизился. Здесь проверяется, даст ли эффект в 2–3 раза больше данных.

**Критерий успеха (зафиксирован до запуска).** На Open Images test средний AUC
«предмет против трудного негатива» выше, чем у `yolo11n_hardneg` (0,751), а recall
на COCO val2017 при консервативном пороге не ниже 82%."""
    + HARDNEG_COMMON_MD,
    "hw3_train_hardneg_v2",
    "yolo11n_hardneg_v2",
    "yolo11n.pt",
)

YOLO11S_CELLS = hardneg_cells(
    """# ДЗ №3. YOLO11s на трудных негативах Open Images

Проверяем, различает ли похожие предметы модель крупнее: YOLO11s (9,4 млн параметров,
21,5 GFLOPs) против YOLO11n (2,6 млн, 6,5 GFLOPs) на тех же данных, что `yolo11n_hardneg_v2`.
Цена — примерно втрое больше вычислений, что важно для работы в браузере в ДЗ №4.

**Критерий успеха (зафиксирован до запуска).** На Open Images test средний AUC
«предмет против трудного негатива» выше, чем у `yolo11n_hardneg_v2`, хотя бы на 0,03;
иначе дополнительная задержка в браузере не оправдана."""
    + HARDNEG_COMMON_MD,
    "hw3_train_yolo11s_hardneg",
    "yolo11s_hardneg",
    "yolo11s.pt",
)

EVAL_CELLS = [
    (
        "markdown",
        """# ДЗ №3. Оценка baseline и основной модели

Сравниваем модели по одному протоколу:

- **YOLOv8n** — предобученный baseline;
- **YOLO11n** — предобученный, показывает вклад архитектуры;
- **YOLO11n-home** — дообученный на 10 домашних классах COCO (`02_train_yolo11n.ipynb`);
- **YOLO11n-hardneg** и **YOLO11n-hardneg-v2** — дообученные с трудными негативами
  Open Images train (`04_train_yolo11n_hardneg.ipynb`; v1 — коммит `41e6245`);
- **YOLO11s-hardneg** — модель крупнее на тех же данных (`05_train_yolo11s_hardneg.ipynb`).

Дообученные модели берутся из входных данных ноутбука: какие подключены, те и оцениваются.

Каждая модель проверяется на размерах входа 640, 416 и 320 и на двух наборах:
COCO val2017 (протокол ДЗ №2, пороги по фолдам) и Open Images V7 test
(перенос на другой источник; пороги взяты с COCO и здесь не подбираются).

**Настройки Kaggle:** GPU T4, Internet — On.
**Входные данные:** COCO 2017 val (как в ДЗ №2) и выводы ноутбуков обучения.""",
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
    copy_images,
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
INPUT_VAL_IMAGES = next(
    p for p in (val_root / "val2017", val_root / "images" / "val2017") if p.is_dir()
)
# Оценка проходит по val много раз (модели × 3 размера), поэтому оригиналы один раз
# копируются на локальный диск без изменений, как в ДЗ №2.
VAL_IMAGES = Path("/tmp/coco_val2017")
failed_copies = copy_images(sorted(INPUT_VAL_IMAGES.glob("*.jpg")), VAL_IMAGES)
assert not failed_copies, failed_copies[:5]
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
        """FINE_TUNED = {}
for name in ("yolo11n_home", "yolo11n_hardneg", "yolo11n_hardneg_v2", "yolo11s_hardneg"):
    found = sorted(Path("/kaggle/input").rglob(f"{name}.pt"))
    if found:
        FINE_TUNED[name] = found[0]
if not FINE_TUNED:
    raise FileNotFoundError("Добавьте во входные данные вывод ноутбука обучения (02 или 04)")
# Разбор ошибок — для последней дообученной модели.
MAIN_MODEL = list(FINE_TUNED)[-1]
MODELS = {
    "yolov8n": get_checkpoint("yolov8n.pt", OUTPUT / "pretrained"),
    "yolo11n": get_checkpoint("yolo11n.pt", OUTPUT / "pretrained"),
    **FINE_TUNED,
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
протоколом `confidence_95pct`. Для дообученных моделей они сохраняются
в `thresholds_<model>_<size>.json`: такой файл в ДЗ №4 загрузит браузер.""",
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
for name in FINE_TUNED:
    for size in IMAGE_SIZES:
        thresholds[(name, size)].to_json(OUTPUT / f"thresholds_{name}_{size}.json")
print((OUTPUT / f"thresholds_{MAIN_MODEL}_640.json").read_text())""",
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
        """pairs = [("yolo11n", "yolov8n")]
for name in FINE_TUNED:
    pairs += [(name, "yolov8n"), (name, "yolo11n")]
for a, b in (
    ("yolo11n_hardneg", "yolo11n_home"),
    ("yolo11n_hardneg_v2", "yolo11n_hardneg"),
    ("yolo11s_hardneg", "yolo11n_hardneg_v2"),
):
    if a in FINE_TUNED and b in FINE_TUNED:
        pairs.append((a, b))
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
    (
        "markdown",
        """## Компромисс на трудных негативах

Порог COCO почти не защищает от похожих предметов. Здесь порог каждого класса
поднимается так, чтобы FAR на трудных негативах Open Images был 40, 20 или 10%;
порог считается на половине фото и проверяется на другой половине (затем половины
меняются местами). Тот же порог применяется к COCO val2017. Порог не опускается ниже
порога COCO. AUC «предмет против трудного негатива» показывает, насколько модель
вообще различает предмет и похожий предмет.""",
    ),
    ("code", TRADEOFF),
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
        """## Разбор ошибок дообученной модели

Для рюкзака, телефона и бутылки берём на Open Images (640) ложное подтверждение
с наибольшим score и пропуск с наименьшим. Авторы и лицензии показанных фото —
в `error_cases.csv`.""",
    ),
    (
        "code",
        """ERROR_TARGETS = ["backpack", "cell phone", "bottle"]
oi_scores = scores[("openimages", MAIN_MODEL, 640)]
records = []
for target in ERROR_TARGETS:
    column = TARGETS.index(target)
    known = oi.mask(column)
    truth = oi.labels[:, column] == 1
    predicted = oi_predictions[(MAIN_MODEL, 640)][target]
    for kind, mask in (
        ("false_positive", predicted & ~truth & known),
        ("false_negative", ~predicted & truth & known),
    ):
        index = np.flatnonzero(mask)
        for row in index[np.argsort(oi_scores[index, column], kind="stable")]:
            records.append({"target": target, "error_type": kind,
                            "image_id": oi.image_ids[row], "score": float(oi_scores[row, column]),
                            "cutoff": thresholds[(MAIN_MODEL, 640)].rules[target].high})
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
    axis.set_title(f"{MAIN_MODEL}: {row.target} — {kind}\\n{row.image_id}, score {row.score:.3f}, "
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
ONNX-версий дообученных моделей с PyTorch на 300 фото COCO val2017.""",
    ),
    (
        "code",
        """rng = np.random.default_rng(SEED)
sample = sorted(rng.choice(len(coco.image_paths), 300, replace=False).tolist())
sample_paths = [coco.image_paths[i] for i in sample]
parity_rows = []
for name, weights in FINE_TUNED.items():
    for size in IMAGE_SIZES:
        onnx_path = weights.parent / f"{name}_{size}.onnx"
        onnx_scores = predict_scores(onnx_path, sample_paths, TARGETS, size, batch_size=1,
                                     device="cpu", **YOLO_SETTINGS)
        torch_scores = scores[("coco", name, size)][sample]
        rule = thresholds[(name, size)]
        cutoffs = np.array([rule.rules[t].high for t in TARGETS])
        parity_rows.append(
            {"model": name, "image_size": size,
             "max_abs_score_diff": float(np.abs(onnx_scores - torch_scores).max()),
             "decision_agreement": float(
                 ((onnx_scores > cutoffs) == (torch_scores > cutoffs)).mean()),
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


COCO_VAL_CELL = next(source for kind, source in EVAL_CELLS if "kagglehub" in source)
OI_DOWNLOAD_CELLS = [
    source
    for kind, source in EVAL_CELLS
    if "test-images-with-rotation" in source or "sample_openimages(csv_paths" in source
]

VERIFY_CELLS = [
    (
        "markdown",
        """# ДЗ №3. Вторая ступень: проверка выреза предмета

Детектор плохо отличает предмет от похожего (AUC «предмет против трудного негатива»
Open Images около 0,75): поднять порог можно только ценой половины правильных находок.
Различение бутылки и банки — задача тонкой классификации, а детектор видит предмет
на всём кадре 640 px. Проверяем двухступенчатую схему без обучения:

1. детектор находит самую уверенную рамку класса;
2. MobileCLIP-S0 (как в ДЗ №2) оценивает **вырез** этой рамки: вероятность подписи
   класса среди подписей класса и четырёх похожих предметов (`photoquest_eval.verify`;
   списки похожих предметов зафиксированы до запуска).

Фото подтверждается, если score детектора выше порога COCO **и** вероятность подписи
выше порога второй ступени. Порог второй ступени подбирается на половине фото Open Images
под FAR на трудных негативах 40/20/10% и проверяется на другой половине, как в `03`.

**Критерий успеха (зафиксирован до запуска).** При FAR на трудных негативах 20%
(вход 640, среднее по классам) двухступенчатая схема сохраняет recall на Open Images
не ниже 70% и на COCO val2017 не ниже 65% (у одного детектора 56% и 47%).

**Настройки Kaggle:** GPU T4, Internet — On.
**Входные данные:** COCO 2017 (val) и вывод ноутбука обучения детектора на трудных негативах.""",
    ),
    ("code", INSTALL),
    ("code", COMMON),
    (
        "code",
        """_ = subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "open_clip_torch==4.2.1", "timm==1.0.31",
     "ftfy==6.3.1"]
)
MOBILECLIP_COMMIT = "48faa0fea4b08d74188b3841771aca6ff2c92852"
_ = subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     f"git+https://github.com/apple/ml-mobileclip.git@{MOBILECLIP_COMMIT}"]
)""",
    ),
    (
        "code",
        """import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from photoquest_eval.calibration import threshold_for_far
from photoquest_eval.data import (
    OPENIMAGES_URLS,
    copy_images,
    download_openimages,
    load_coco,
    load_openimages,
    sample_openimages,
)
from photoquest_eval.inference import predict_best_boxes
from photoquest_eval.tradeoff import mean_rates, status_rates, two_fold_tradeoff
from photoquest_eval.verify import LOOKALIKES, crop_box, prompts, target_probability

OUTPUT = Path("/kaggle/working/hw3_verify")
OUTPUT.mkdir(parents=True, exist_ok=True)
OI_SAMPLE = {"positives_per_class": 400, "negatives_per_class": 800, "seed": SEED}
IMAGE_SIZE = 640
HN_LEVELS = (0.40, 0.20, 0.10)
CROP_MARGIN = 0.1
MOBILECLIP = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt",
    "809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7",
)""",
    ),
    ("markdown", "## Данные — те же выборки, что в `03_evaluate_models.ipynb`"),
    ("code", COCO_VAL_CELL),
    *[("code", source) for source in OI_DOWNLOAD_CELLS],
    ("markdown", "## Первая ступень: детектор"),
    (
        "code",
        """DETECTOR = None
for name in ("yolo11s_hardneg", "yolo11n_hardneg_v2", "yolo11n_hardneg"):
    found = sorted(Path("/kaggle/input").rglob(f"{name}.pt"))
    if found:
        DETECTOR, DETECTOR_PATH = name, found[0]
        break
if DETECTOR is None:
    raise FileNotFoundError("Добавьте во входные данные вывод ноутбука 04 или 05")
print("Детектор:", DETECTOR, sha256(DETECTOR_PATH))
det = {}
for dataset in (coco, oi):
    started = time.perf_counter()
    det[dataset.name] = predict_best_boxes(
        DETECTOR_PATH, dataset.image_paths, TARGETS, IMAGE_SIZE, device=0, **YOLO_SETTINGS
    )
    print(dataset.name, f"{time.perf_counter() - started:.0f} с")
floors = []
for column in range(len(TARGETS)):
    negatives = det["coco"][0][coco.labels[:, column] == 0, column]
    floors.append(threshold_for_far(negatives, "confidence_95pct", 0.05, 0.95))
pd.Series(floors, index=TARGETS, name="coco_cutoff")""",
    ),
    ("markdown", "## Вторая ступень: MobileCLIP по вырезам"),
    (
        "code",
        """import mobileclip
from torch.utils.data import DataLoader, Dataset

clip_path = OUTPUT / "pretrained" / "mobileclip_s0.pt"
clip_path.parent.mkdir(parents=True, exist_ok=True)
if not clip_path.exists():
    urllib.request.urlretrieve(MOBILECLIP[0], clip_path)
assert sha256(clip_path) == MOBILECLIP[1], "Не совпал SHA-256 MobileCLIP-S0"
clip, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained=str(clip_path), reparameterize=True, device="cuda"
)
clip.eval()
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
with torch.inference_mode():
    text_features = {
        target: F.normalize(clip.encode_text(tokenizer(prompts(target)).to("cuda")).float(), dim=-1)
        for target in TARGETS
    }


class Crops(Dataset):
    \"\"\"Вырезы лучших рамок: (фото, класс) для всех пар, где детектор что-то нашёл.\"\"\"

    def __init__(self, paths, boxes):
        self.paths, self.boxes = paths, boxes
        self.pairs = [
            (row, column)
            for row in range(len(paths))
            for column in range(len(TARGETS))
            if np.isfinite(boxes[row, column]).all()
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row, column = self.pairs[index]
        with Image.open(self.paths[row]) as image:
            image = image.convert("RGB")
            area = crop_box(image.width, image.height, self.boxes[row, column], CROP_MARGIN)
            crop = image.crop(area) if area else image
        return preprocess(crop), row, column


def verify(dataset):
    boxes = det[dataset.name][1]
    probability = np.zeros(boxes.shape[:2], dtype=np.float32)
    loader = DataLoader(Crops(dataset.image_paths, boxes), batch_size=256, num_workers=4)
    for batch, rows, columns in loader:
        with torch.inference_mode():
            features = F.normalize(clip.encode_image(batch.to("cuda")).float(), dim=-1)
        for feature, row, column in zip(features, rows.tolist(), columns.tolist(), strict=True):
            target = TARGETS[column]
            similarity = (feature @ text_features[target].T).cpu().numpy()
            probability[row, column] = target_probability(similarity)
    return probability


clip_scores = {}
clip_timing = {}
for dataset in (coco, oi):
    started = time.perf_counter()
    clip_scores[dataset.name] = verify(dataset)
    clip_timing[dataset.name] = time.perf_counter() - started
    print(dataset.name, f"{clip_timing[dataset.name]:.0f} с")
np.savez_compressed(
    OUTPUT / "two_stage_scores.npz",
    **{f"{name}__detector": det[name][0] for name in det},
    **{f"{name}__boxes": det[name][1] for name in det},
    **{f"{name}__clip": clip_scores[name] for name in clip_scores},
)""",
    ),
    (
        "markdown",
        """## Результат

Для каждого класса сравниваются три правила при одном и том же целевом FAR
на трудных негативах: только детектор; только вторая ступень (среди фото,
где детектор выше порога COCO); их произведение как единый score.""",
    ),
    (
        "code",
        """fold = np.random.default_rng(SEED).random(len(oi.image_ids)) < 0.5
rows, status_records = [], []
for column, target in enumerate(TARGETS):
    labels = oi.labels[:, column]
    known = labels >= 0
    floor = floors[column]
    rules = {
        "detector": (det["openimages"][0][:, column], det["coco"][0][:, column], floor),
        # Вторая ступень работает только на фото, прошедших порог детектора.
        "two_stage": (
            np.where(det["openimages"][0][:, column] > floor, clip_scores["openimages"][:, column], -1.0),
            np.where(det["coco"][0][:, column] > floor, clip_scores["coco"][:, column], -1.0),
            -0.5,
        ),
        "product": (
            det["openimages"][0][:, column] * clip_scores["openimages"][:, column],
            det["coco"][0][:, column] * clip_scores["coco"][:, column],
            0.0,
        ),
    }
    for rule, (oi_s, coco_s, rule_floor) in rules.items():
        row = {"target": target, "rule": rule, "oi_auc_hardneg": roc_auc_score(labels[known], oi_s[known])}
        for point in two_fold_tradeoff(labels, oi_s, rule_floor, HN_LEVELS, fold):
            tag = int(point.level * 100)
            coco_recall, coco_far = mean_rates(coco.labels[:, column], coco_s, point.cutoffs)
            row |= {f"oi_recall@hn{tag}": point.recall, f"oi_far@hn{tag}": point.far,
                    f"coco_recall@hn{tag}": coco_recall, f"coco_far@hn{tag}": coco_far}
        rows.append(row)
    # Три статуса двухступенчатой схемы: «да» — выше порога HN 20%, «не уверен» — прошёл
    # только детектор, «не найдено» — детектор ниже порога COCO.
    oi_s, coco_s, _ = rules["two_stage"]
    point = two_fold_tradeoff(labels, oi_s, -0.5, (0.20,), fold)[0]
    high = float(np.mean(point.cutoffs))
    for name, values in (
        ("oi_positive", oi_s[labels == 1]),
        ("oi_hardneg", oi_s[labels == 0]),
        ("coco_positive", coco_s[coco.labels[:, column] == 1]),
        ("coco_negative", coco_s[coco.labels[:, column] == 0]),
    ):
        confirmed, uncertain, not_found = status_rates(values, -0.5, high)
        status_records.append({"target": target, "set": name, "confirmed": confirmed,
                               "uncertain": uncertain, "not_found": not_found})
two_stage = pd.DataFrame(rows)
two_stage.to_csv(OUTPUT / "two_stage_tradeoff_per_class.csv", index=False)
statuses = pd.DataFrame(status_records)
statuses.to_csv(OUTPUT / "two_stage_statuses_per_class.csv", index=False)
display(two_stage.groupby("rule", sort=False).mean(numeric_only=True).round(3).T)
display(two_stage.pivot(index="target", columns="rule", values="oi_auc_hardneg").round(3))
display(two_stage.pivot(index="target", columns="rule", values="oi_recall@hn20").round(3))
display(statuses.groupby("set").mean(numeric_only=True).round(3))""",
    ),
    (
        "code",
        """summary = two_stage.groupby("rule").mean(numeric_only=True)
criterion = {
    "oi_recall@hn20": float(summary.loc["two_stage", "oi_recall@hn20"]),
    "coco_recall@hn20": float(summary.loc["two_stage", "coco_recall@hn20"]),
}
criterion["passed"] = criterion["oi_recall@hn20"] >= 0.70 and criterion["coco_recall@hn20"] >= 0.65
crops = {name: int(np.isfinite(det[name][1]).all(axis=-1).sum()) for name in det}
metadata = {
    "repo_ref": REPO_REF,
    "repo_commit": REPO_COMMIT,
    "detector": DETECTOR,
    "detector_sha256": sha256(DETECTOR_PATH),
    "mobileclip_sha256": MOBILECLIP[1],
    "mobileclip_commit": MOBILECLIP_COMMIT,
    "lookalikes": {target: list(LOOKALIKES[target]) for target in TARGETS},
    "crop_margin": CROP_MARGIN,
    "image_size": IMAGE_SIZE,
    "hn_levels": list(HN_LEVELS),
    "criterion": criterion,
    "crops": crops,
    "clip_seconds": clip_timing,
    "clip_ms_per_crop_t4": {name: 1000 * clip_timing[name] / max(crops[name], 1) for name in crops},
    "gpu": torch.cuda.get_device_name(0),
    "python": platform.python_version(),
    "packages": package_versions("torch", "ultralytics", "open_clip_torch", "timm", "numpy"),
    "started_utc": STARTED_UTC,
    "elapsed_seconds": round(time.perf_counter() - STARTED, 1),
}
(OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
with zipfile.ZipFile("/kaggle/working/hw3_verify_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
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
    build(HARDNEG_V2_CELLS, HERE / "04_train_yolo11n_hardneg.ipynb")
    build(YOLO11S_CELLS, HERE / "05_train_yolo11s_hardneg.ipynb")
    build(VERIFY_CELLS, HERE / "06_crop_verification.ipynb")
