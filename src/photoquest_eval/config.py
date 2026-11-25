"""Общие константы эксперимента ДЗ №3.

Значения совпадают с протоколом ДЗ №2, чтобы результаты были сопоставимы.
"""

SEED = 2026
CV_SPLITS = 5
FAR_TARGET = 0.05
CONFIDENCE = 0.95
PROTOCOLS = ("empirical_5pct", "confidence_95pct")

# Десять домашних классов первого PoC. Ключ — имя класса COCO,
# значение — имя boxable-класса Open Images V7 (проверено по
# oidv7-class-descriptions-boxable.csv).
HOME_CLASSES = {
    "bottle": "Bottle",
    "cell phone": "Mobile phone",
    "backpack": "Backpack",
    "clock": "Clock",
    "vase": "Vase",
    "potted plant": "Houseplant",
    "laptop": "Laptop",
    "keyboard": "Computer keyboard",
    "tv": "Television",
    "banana": "Banana",
}
TARGETS = tuple(HOME_CLASSES)

# Параметры детектора из ДЗ №2. В браузере (ДЗ №4) размер входа может быть меньше,
# поэтому оценка повторяется для каждого значения из IMAGE_SIZES.
YOLO_SETTINGS = {"conf": 0.001, "iou": 0.7, "max_det": 300}
IMAGE_SIZES = (640, 416, 320)

# Срез «основная находка / фон». Порог фиксируется до запуска моделей.
# Основная находка: рамка занимает не меньше MAIN_MIN_AREA площади кадра
# и её центр лежит в центральной зоне шириной и высотой MAIN_CENTER_ZONE кадра.
MAIN_MIN_AREA = 0.05
MAIN_CENTER_ZONE = 0.6

# Мелкий объект: рамка меньше 1% площади кадра (как в ДЗ №2).
SMALL_OBJECT_AREA = 0.01
