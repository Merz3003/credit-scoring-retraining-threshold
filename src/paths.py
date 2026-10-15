"""
Пути проекта.

Все каталоги задаются относительно корня репозитория и создаются при импорте.
Переопределить расположение можно переменными окружения:

    SCORING_DATA     исходные файлы Lending Club (по умолчанию data/)
    SCORING_WORK     промежуточные файлы пайплайна (по умолчанию work/)
    SCORING_RESULTS  итоговые CSV (по умолчанию results/)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = Path(os.environ.get("SCORING_DATA", ROOT / "data"))
WORK = Path(os.environ.get("SCORING_WORK", ROOT / "work"))
RESULTS = Path(os.environ.get("SCORING_RESULTS", ROOT / "results"))

for d in (DATA, WORK, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# Исходные данные: скачиваются с Kaggle, в репозиторий не входят.
ACCEPTED = DATA / "accepted_2007_to_2018Q4_csv.gz"
REJECTED_ZIP = DATA / "rejected_2007_to_2018Q4_csv.zip"

# Промежуточные файлы.
PANEL = WORK / "panel_ext.parquet"
REJECTED = WORK / "rej.pkl"


def cache(train_months: int) -> Path:
    """Кэш предсказаний для заданной длины окна обучения."""
    return WORK / f"cache_tw{train_months}.npz"


def require(path: Path) -> Path:
    """Проверяет, что файл на месте, и объясняет, что делать, если нет."""
    if not path.exists():
        if path.parent == DATA:
            raise SystemExit(
                f"Нет файла {path.name} в {DATA}.\n"
                "Скачайте набор Lending Club с "
                "https://www.kaggle.com/datasets/wordsforthewise/lending-club "
                f"и положите оба архива в {DATA}."
            )
        raise SystemExit(
            f"Нет файла {path.name} в {path.parent}.\n"
            "Запустите предыдущие шаги пайплайна (см. README, раздел «Воспроизведение»)."
        )
    return path
