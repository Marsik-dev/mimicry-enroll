"""
Детектор эмоций на основе MediaPipe FaceLandmarker blendshapes.

Использует уже установленный mimicry_preproc для пути к модели face_landmarker.task
и стандартный mediapipe API. Возвращает одну из 5 эмоций:
happy, angry, surprise, sad, disgust — или None если правило не сработало.

Подтверждение эмоции: текущий кадр + хранение истории через `update()`,
эмоция считается стабильной если N кадров подряд её предсказывают.
"""
from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)

SUPPORTED_EMOTIONS = ("happy", "angry", "surprise", "sad", "disgust")

# Confirmation: эмоция считается подтверждённой если она держится >= STABILITY_FRAMES кадров
STABILITY_FRAMES = 8


def _get_model_path() -> str:
    from mimicry_preproc.stages import landmark_extractor as le
    return str(le._DEFAULT_MODEL)


def _bs(blendshapes, name: str) -> float:
    """Получить значение конкретного blendshape по имени."""
    for cat in blendshapes:
        if cat.category_name == name:
            return cat.score
    return 0.0


def classify_blendshapes(blendshapes) -> str | None:
    """Эмпирические правила: blendshapes → emotion. Возвращает None если правил не сработало."""
    smile = _bs(blendshapes, "mouthSmileLeft") + _bs(blendshapes, "mouthSmileRight")
    frown = _bs(blendshapes, "mouthFrownLeft") + _bs(blendshapes, "mouthFrownRight")
    brow_down = _bs(blendshapes, "browDownLeft") + _bs(blendshapes, "browDownRight")
    brow_inner_up = _bs(blendshapes, "browInnerUp")
    brow_outer_up = _bs(blendshapes, "browOuterUpLeft") + _bs(blendshapes, "browOuterUpRight")
    eye_squint = _bs(blendshapes, "eyeSquintLeft") + _bs(blendshapes, "eyeSquintRight")
    jaw_open = _bs(blendshapes, "jawOpen")
    nose_sneer = _bs(blendshapes, "noseSneerLeft") + _bs(blendshapes, "noseSneerRight")
    mouth_upper_up = _bs(blendshapes, "mouthUpperUpLeft") + _bs(blendshapes, "mouthUpperUpRight")

    # Priority order: проверяем более характерные эмоции первыми
    if smile > 0.5 and frown < 0.15:
        return "happy"
    if (brow_inner_up + brow_outer_up) > 0.6 and jaw_open > 0.25:
        return "surprise"
    if brow_down > 0.55 and eye_squint > 0.3:
        return "angry"
    if nose_sneer > 0.4 or (mouth_upper_up > 0.4 and brow_down > 0.2):
        return "disgust"
    if frown > 0.3 and brow_inner_up > 0.15 and smile < 0.1:
        return "sad"
    return None


class EmotionDetector:
    """Детектор эмоций по одному BGR-кадру."""

    def __init__(self) -> None:
        import mediapipe as mp
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_get_model_path()),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._history: deque[str | None] = deque(maxlen=STABILITY_FRAMES)

    def detect(self, frame_bgr: np.ndarray) -> str | None:
        """Сырое распознавание одного кадра. Возвращает эмоцию или None."""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_blendshapes:
            return None
        return classify_blendshapes(result.face_blendshapes[0])

    def update(self, frame_bgr: np.ndarray) -> tuple[str | None, str | None]:
        """Обновляет историю кадров. Возвращает (current_emotion, confirmed_emotion).

        confirmed_emotion: одна и та же эмоция STABILITY_FRAMES кадров подряд.
        """
        current = self.detect(frame_bgr)
        self._history.append(current)

        confirmed: str | None = None
        if len(self._history) == STABILITY_FRAMES:
            first = self._history[0]
            if first is not None and all(x == first for x in self._history):
                confirmed = first

        return current, confirmed

    def reset(self) -> None:
        self._history.clear()

    def close(self) -> None:
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
