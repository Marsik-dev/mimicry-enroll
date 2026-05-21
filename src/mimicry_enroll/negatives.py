"""
Загрузка чужих векторов признаков из MEAD-датасета для НПБК-обучения.

При первом обращении скачивает архив с GitHub Release и кэширует в файловой системе.
В памяти процесса векторы кэшируются после первой загрузки.
"""
from __future__ import annotations

import logging
import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

log = logging.getLogger(__name__)

MEAD_RELEASE_URL = (
    "https://github.com/Marsik-dev/datasets/releases/download/"
    "v1.0.0/mead_cache.tar.gz"
)

NEGATIVES_DIR = Path(os.environ.get("NEGATIVES_DIR", "/data/negatives"))

_LOADED_VECTORS: dict[str, list[np.ndarray]] | None = None


def _ensure_dataset() -> Path:
    """Скачивает и распаковывает MEAD-кэш если ещё не на диске. Возвращает путь к mead_cache/."""
    cache_root = NEGATIVES_DIR / "mead_cache"
    if cache_root.exists() and any(cache_root.iterdir()):
        return cache_root

    NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)
    archive = NEGATIVES_DIR / "mead_cache.tar.gz"

    if not archive.exists():
        log.info("Скачиваю MEAD-кэш из %s", MEAD_RELEASE_URL)
        urlretrieve(MEAD_RELEASE_URL, archive)
        log.info("Скачано %.1f МБ", archive.stat().st_size / 1e6)

    log.info("Распаковываю архив в %s", NEGATIVES_DIR)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(NEGATIVES_DIR)
    archive.unlink(missing_ok=True)

    if not cache_root.exists():
        raise RuntimeError(f"После распаковки не найден {cache_root}")
    return cache_root


def _load_all_vectors() -> dict[str, list[np.ndarray]]:
    """Загружает все векторы из MEAD-кэша, сгруппированные по эмоциям."""
    cache_root = _ensure_dataset()
    by_emotion: dict[str, list[np.ndarray]] = {}

    for actor_dir in sorted(cache_root.iterdir()):
        if not actor_dir.is_dir():
            continue
        for emotion_dir in actor_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            emotion = emotion_dir.name
            for npy_file in emotion_dir.glob("*.npy"):
                try:
                    vec = np.load(npy_file)
                    if vec.ndim == 1 and vec.size > 0:
                        by_emotion.setdefault(emotion, []).append(vec)
                except Exception as e:
                    log.warning("Не удалось загрузить %s: %s", npy_file, e)

    log.info(
        "Загружено векторов: %s",
        ", ".join(f"{e}={len(v)}" for e, v in sorted(by_emotion.items())),
    )
    return by_emotion


def _get_cache() -> dict[str, list[np.ndarray]]:
    global _LOADED_VECTORS
    if _LOADED_VECTORS is None:
        _LOADED_VECTORS = _load_all_vectors()
    return _LOADED_VECTORS


def get_negatives_for_emotion(emotion: str, max_n: int = 600) -> list[np.ndarray]:
    """Векторы чужих акторов в указанной эмоции из MEAD.

    Возвращает до `max_n` векторов (перемешанных). Если эмоции в датасете нет,
    fallback на ближайшую (happy если нет, иначе любую доступную).
    """
    cache = _get_cache()
    if emotion in cache and cache[emotion]:
        pool = cache[emotion]
    elif "happy" in cache and cache["happy"]:
        log.warning("Эмоция '%s' не найдена в MEAD, использую happy", emotion)
        pool = cache["happy"]
    elif cache:
        any_emotion = next(iter(cache))
        log.warning("Эмоция '%s' не найдена, использую %s", emotion, any_emotion)
        pool = cache[any_emotion]
    else:
        raise RuntimeError("MEAD-кэш пустой")

    if len(pool) <= max_n:
        return list(pool)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pool), size=max_n, replace=False)
    return [pool[i] for i in idx]
