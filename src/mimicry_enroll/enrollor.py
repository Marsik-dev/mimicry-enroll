"""
Enrollment pipeline: video → feature vectors → НПБК training → SSH keygen.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

log = logging.getLogger(__name__)

_PIPELINE = None
_PIPELINE_CFG = None


def _get_pipeline():
    global _PIPELINE, _PIPELINE_CFG
    if _PIPELINE is None:
        from mimicry_preproc.pipeline import Pipeline, PipelineConfig
        from mimicry_preproc.stages.quality_filter import QualityFilterConfig
        cfg = PipelineConfig.default()
        cfg.quality_filter = QualityFilterConfig(
            min_sharpness=15.0, min_brightness=0.1, max_brightness=0.95, min_overall=0.4
        )
        _PIPELINE = Pipeline(cfg)
        _PIPELINE_CFG = cfg
    return _PIPELINE


@dataclass
class EnrollResult:
    container_bytes: bytes        # NBKContainer.to_bytes()
    reference_code: np.ndarray    # stable binary code for encryption
    public_key_text: str          # SSH public key (OpenSSH line)
    private_key_pem: bytes        # SSH private key PEM (for encryption)
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


def train_npbk(own_vectors: list[np.ndarray], other_vectors: list[np.ndarray] | None = None):
    """Обучить НПБК классификатор. Возвращает NBKContainer."""
    from npbk import NBKContainer, NPBKTrainer, TrainingSet
    from npbk.augmentation import augment_own

    own = np.stack(own_vectors)
    aug = augment_own(own)
    own_aug = np.vstack([own, aug])

    if other_vectors:
        other = np.stack(other_vectors[:600])
    else:
        # fallback: shuffle augmented own as pseudo-other (suboptimal)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(own_aug))
        other = own_aug[idx[:max(64, len(own_aug) // 2)]]

    ts = TrainingSet(own_vectors=own_aug, other_vectors=other)
    result = NPBKTrainer().train(ts)
    return NBKContainer.from_result(result, feature_dim=own.shape[1])


def generate_ssh_keypair() -> tuple[bytes, str]:
    """Генерирует Ed25519 SSH ключи. Возвращает (private_pem, public_openssh_line)."""
    private_key = Ed25519PrivateKey.generate()
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    )
    return pem, pub.decode()


def enroll(
    video_paths: list[Path],
    uid: str,
    extra_other_vectors: list[np.ndarray] | None = None,
) -> EnrollResult:
    """
    Полный enrollment pipeline.
    1. Извлечь векторы из видео
    2. Обучить НПБК
    3. Сгенерировать SSH ключи
    """
    log.info("[%s] Enrollment: %d видео", uid, len(video_paths))

    own_vectors = extract_vectors_from_videos(video_paths)
    if len(own_vectors) < 3:
        raise ValueError(f"Слишком мало качественных фреймов ({len(own_vectors)}). Нужно ≥3.")

    log.info("[%s] Векторов извлечено: %d", uid, len(own_vectors))

    container = train_npbk(own_vectors, extra_other_vectors)
    log.info("[%s] НПБК обучен: code_length=%d stability=%.3f",
             uid, container.quality.code_length if hasattr(container, 'quality') else 0,
             container.quality.mean_stability if hasattr(container, 'quality') else 0.0)

    private_pem, public_key_text = generate_ssh_keypair()

    # Extract stats
    q = getattr(container, "quality", None)
    mean_stability = q.mean_stability if q else 0.0
    code_length = len(container.reference_code) if hasattr(container, "reference_code") else 0
    warnings_list = (q.warnings if q else []) or []

    return EnrollResult(
        container_bytes=container.to_bytes(),
        reference_code=container.reference_code,
        public_key_text=public_key_text,
        private_key_pem=private_pem,
        n_vectors=len(own_vectors),
        mean_stability=mean_stability,
        code_length=code_length,
        warnings=warnings_list,
    )
