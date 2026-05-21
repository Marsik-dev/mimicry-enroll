"""
Enrollment pipeline: video → feature vectors → НПБК training → SSH keygen/import.

НПБК схема (персона+эмоция):
  own   = свои векторы в основной эмоции
  other = свои векторы в других эмоциях + чужие акторы в основной эмоции (из MEAD)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key as rsa_generate

log = logging.getLogger(__name__)

KeyType = Literal["ed25519", "rsa", "imported"]

_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from mimicry_preproc.pipeline import Pipeline, PipelineConfig
        from mimicry_preproc.stages.quality_filter import QualityFilterConfig
        cfg = PipelineConfig.default()
        cfg.quality_filter = QualityFilterConfig(
            min_sharpness=15.0, min_brightness=0.1, max_brightness=0.95, min_overall=0.4
        )
        _PIPELINE = Pipeline(cfg)
    return _PIPELINE


@dataclass
class EnrollResult:
    container_bytes: bytes
    reference_code: np.ndarray
    public_key_text: str
    private_key_pem: bytes
    n_vectors: int
    mean_stability: float
    code_length: int
    warnings: list[str]


def extract_vectors_from_videos(video_paths: list[Path]) -> list[np.ndarray]:
    """Запустить Pipeline на каждом видео, вернуть список векторов признаков."""
    pipeline = _get_pipeline()
    vectors = []
    for vp in video_paths:
        try:
            fv = pipeline.run(str(vp))
            if fv is not None and fv.combined.size > 0:
                vectors.append(fv.combined)
                log.info("  %s → вектор dim=%d", vp.name, fv.combined.shape[0])
            else:
                log.warning("  %s → пустой вектор", vp.name)
        except Exception as e:
            log.warning("  %s → ошибка Pipeline: %s", vp.name, e)
    return vectors


def train_npbk(
    own_vectors: list[np.ndarray],
    own_other_emotion_vectors: list[np.ndarray] | None = None,
    foreign_vectors: list[np.ndarray] | None = None,
):
    """Обучить НПБК. other = свои в других эмоциях + чужие в основной эмоции."""
    from npbk import NBKContainer, NPBKTrainer, TrainingSet
    from npbk.augmentation import augment_own

    own = np.stack(own_vectors)
    # Аугментируем до 300 — стандарт для НПБК-обучения (см. evaluate_emotion.py / mixed).
    target = max(300, len(own))
    own_aug = augment_own(own, target_count=target)

    other_list: list[np.ndarray] = []
    if own_other_emotion_vectors:
        other_list.extend(own_other_emotion_vectors)
    if foreign_vectors:
        remaining = max(600 - len(other_list), 0)
        other_list.extend(foreign_vectors[:remaining])

    if other_list:
        other = np.stack(other_list)
    else:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(own_aug))
        other = own_aug[idx[: max(64, len(own_aug) // 2)]]

    ts = TrainingSet(own_vectors=own_aug, other_vectors=other)
    result = NPBKTrainer().train(ts)
    return NBKContainer.from_result(result, feature_dim=own.shape[1])


def _pub_openssh(private_key) -> str:
    pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    )
    return pub.decode()


def _priv_openssh_pem(private_key) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )


def generate_keypair(
    key_type: KeyType = "ed25519",
    imported_pem: bytes | None = None,
) -> tuple[bytes, str]:
    """Возвращает (private_pem, public_openssh_line) для указанного типа ключа."""
    if key_type == "ed25519":
        priv = Ed25519PrivateKey.generate()
        return _priv_openssh_pem(priv), _pub_openssh(priv)
    if key_type == "rsa":
        priv = rsa_generate(public_exponent=65537, key_size=4096)
        return _priv_openssh_pem(priv), _pub_openssh(priv)
    if key_type == "imported":
        if not imported_pem:
            raise ValueError("imported_pem обязателен для key_type='imported'")
        priv = serialization.load_ssh_private_key(imported_pem, password=None)
        return _priv_openssh_pem(priv), _pub_openssh(priv)
    raise ValueError(f"Unknown key_type: {key_type}")


def enroll(
    main_emotion_vectors: list[np.ndarray],
    other_emotion_vectors: list[np.ndarray] | None = None,
    main_emotion: str = "happy",
    key_type: KeyType = "ed25519",
    imported_pem: bytes | None = None,
) -> EnrollResult:
    """
    Полный enrollment.

    main_emotion_vectors  — векторы пользователя в основной эмоции (≥3)
    other_emotion_vectors — векторы пользователя в других эмоциях (для дискриминации)
    main_emotion          — название основной эмоции (для подгрузки negatives)
    """
    if len(main_emotion_vectors) < 3:
        raise ValueError(
            f"Слишком мало векторов основной эмоции ({len(main_emotion_vectors)}). Нужно ≥3."
        )

    log.info(
        "Enrollment: main=%d other=%d emotion=%s key_type=%s",
        len(main_emotion_vectors),
        len(other_emotion_vectors or []),
        main_emotion,
        key_type,
    )

    from mimicry_enroll.negatives import get_negatives_for_emotion
    try:
        foreign = get_negatives_for_emotion(main_emotion, max_n=500)
        log.info("Загружено чужих векторов: %d", len(foreign))
    except Exception as e:
        log.warning("Не удалось загрузить чужих: %s", e)
        foreign = []

    container = train_npbk(
        own_vectors=main_emotion_vectors,
        own_other_emotion_vectors=other_emotion_vectors,
        foreign_vectors=foreign,
    )

    code_length = (
        len(container.reference_code)
        if hasattr(container, "reference_code")
        else 0
    )
    q = getattr(container, "quality", None)
    mean_stability = q.mean_stability if q else 0.0
    warnings_list = (q.warnings if q else []) or []

    log.info(
        "НПБК обучен: code_length=%d stability=%.3f",
        code_length, mean_stability,
    )

    private_pem, public_key_text = generate_keypair(key_type, imported_pem)

    return EnrollResult(
        container_bytes=container.to_bytes(),
        reference_code=container.reference_code,
        public_key_text=public_key_text,
        private_key_pem=private_pem,
        n_vectors=len(main_emotion_vectors),
        mean_stability=mean_stability,
        code_length=code_length,
        warnings=warnings_list,
    )
