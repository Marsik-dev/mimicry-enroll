"""Страница регистрации: WebRTC wizard с детекцией эмоций в реальном времени."""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import threading
from collections import deque
from pathlib import Path

import av
import cv2
import httpx
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

log = logging.getLogger(__name__)

EMOTION_EMOJI = {
    "happy": "😊",
    "angry": "😠",
    "surprise": "😲",
    "sad": "😢",
}
EMOTION_LABELS = {
    "happy": "😊 happy (улыбка)",
    "angry": "😠 angry (нахмуренный)",
    "surprise": "😲 surprise (удивление)",
    "sad": "😢 sad (грусть)",
}

API_URL = os.environ.get("INTERNAL_API_URL", "http://localhost:8000")
FRAGMENT_FPS = 24
FRAGMENT_SECONDS = 3
FRAGMENT_BUFFER = FRAGMENT_FPS * FRAGMENT_SECONDS
COOLDOWN_FRAMES = FRAGMENT_FPS * 2  # 2 сек паузы после захвата фрагмента

# WebRTC: явные STUN сервера для надёжности при VPN/мобильных сетях.
# Для чистого LAN можно поставить iceServers=[] — только host-кандидаты.
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
    ],
    "iceTransportPolicy": "all",
}


class EnrollProcessor(VideoProcessorBase):
    """Видео-процессор: детектит эмоции, при подтверждении пишет фрагмент и загружает."""

    def __init__(self) -> None:
        self.detector = None
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=FRAGMENT_BUFFER)
        self.lock = threading.Lock()
        self.session_id: str | None = None
        self.schedule: dict[str, int] = {}
        self.uploaded_counts: dict[str, int] = {}
        self.current_target: str | None = None
        self.last_detected: str | None = None
        self.cooldown: int = 0
        self.error_message: str | None = None

    def _ensure_detector(self):
        if self.detector is None:
            from mimicry_enroll.emotion_detector import EmotionDetector
            self.detector = EmotionDetector()

    def _next_target(self) -> str | None:
        for emo, target in self.schedule.items():
            if self.uploaded_counts.get(emo, 0) < target:
                return emo
        return None

    def _save_and_upload(self, emotion: str, frames: list[np.ndarray]) -> None:
        """Сохранить фрагмент в mp4 и отправить на /api/enroll/upload-fragment."""
        if not frames or not self.session_id:
            return
        h, w = frames[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            path = Path(tf.name)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, FRAGMENT_FPS, (w, h))
            try:
                for f in frames:
                    writer.write(f)
            finally:
                writer.release()

            with open(path, "rb") as fp:
                files = {"video": (path.name, fp, "video/mp4")}
                data = {"session_id": self.session_id, "emotion": emotion}
                try:
                    r = httpx.post(
                        f"{API_URL}/api/enroll/upload-fragment",
                        files=files,
                        data=data,
                        timeout=30,
                    )
                    r.raise_for_status()
                    progress = r.json().get("progress", {})
                    with self.lock:
                        for e, p in progress.items():
                            self.uploaded_counts[e] = p.get("current", 0)
                except Exception as e:
                    self.error_message = f"Upload error: {e}"
                    log.exception("Failed to upload fragment")
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._ensure_detector()

        with self.lock:
            self.frame_buffer.append(img.copy())
            target = self.current_target
            cooldown = self.cooldown

        current, confirmed = self.detector.update(img) if self.detector else (None, None)
        self.last_detected = current

        if cooldown > 0:
            with self.lock:
                self.cooldown = cooldown - 1
        elif confirmed and confirmed == target:
            with self.lock:
                frames_copy = list(self.frame_buffer)
                self.cooldown = COOLDOWN_FRAMES
            self.detector.reset()
            threading.Thread(
                target=self._save_and_upload,
                args=(target, frames_copy),
                daemon=True,
            ).start()

        # overlay
        if target:
            color = (0, 200, 0) if current == target else (180, 180, 180)
            text = f"{EMOTION_EMOJI.get(target, '?')} {target}: {current or '...'}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            done = self.uploaded_counts.get(target, 0)
            need = self.schedule.get(target, 0)
            cv2.putText(
                img, f"{done}/{need}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────────────────────────────────────
#  Streamlit page
# ──────────────────────────────────────────────────────────────────────────────


def _reset_state() -> None:
    for k in (
        "enroll_step", "session_id", "schedule", "main_emotion",
        "display_name", "key_type", "imported_pem", "enroll_result",
    ):
        st.session_state.pop(k, None)


def _step_settings() -> None:
    st.subheader("Шаг 1: настройки")

    display_name = st.text_input("Имя пользователя", max_chars=64, placeholder="Alice")
    main_emotion = st.selectbox(
        "Основная эмоция (используется для входа)",
        list(EMOTION_LABELS.keys()),
        format_func=lambda e: EMOTION_LABELS[e],
    )

    st.markdown("**Тип SSH-ключа**")
    key_type = st.radio(
        "key_type",
        ["ed25519", "rsa", "imported"],
        format_func=lambda x: {
            "ed25519": "🔑 Сгенерировать Ed25519 (рекомендуется)",
            "rsa": "🔑 Сгенерировать RSA-4096",
            "imported": "📤 Импортировать свой private key",
        }[x],
        label_visibility="collapsed",
    )

    imported_pem_b64 = None
    if key_type == "imported":
        pem_text = st.text_area(
            "Вставьте OpenSSH PEM (начинается с -----BEGIN OPENSSH PRIVATE KEY-----)",
            height=200,
        )
        if pem_text.strip():
            imported_pem_b64 = base64.b64encode(pem_text.encode()).decode()

    if st.button("Далее ▶", type="primary", use_container_width=True):
        if not display_name.strip():
            st.error("Введите имя пользователя")
            return
        if key_type == "imported" and not imported_pem_b64:
            st.error("Вставьте PEM ключ")
            return
        try:
            r = httpx.post(
                f"{API_URL}/api/enroll/start",
                json={
                    "display_name": display_name.strip(),
                    "main_emotion": main_emotion,
                    "key_type": key_type,
                    "imported_pem_b64": imported_pem_b64,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Ошибка запуска сессии: {e}")
            return

        st.session_state.enroll_step = 2
        st.session_state.session_id = data["session_id"]
        st.session_state.schedule = data["schedule"]
        st.session_state.main_emotion = main_emotion
        st.session_state.display_name = display_name.strip()
        st.session_state.key_type = key_type
        st.rerun()


def _step_recording() -> None:
    st.subheader("Шаг 2: запись эмоций с камеры")

    session_id = st.session_state.session_id
    schedule: list[dict] = st.session_state.schedule
    schedule_dict: dict[str, int] = {s["emotion"]: s["count"] for s in schedule}

    try:
        r = httpx.get(
            f"{API_URL}/api/enroll/progress",
            params={"session_id": session_id},
            timeout=5,
        )
        r.raise_for_status()
        progress = r.json()
    except Exception:
        progress = {"progress": {e: {"current": 0, "target": n} for e, n in schedule_dict.items()},
                    "is_complete": False}

    # Автообновление пока запись не завершена: дёргаем /progress каждые 2 сек,
    # чтобы UI отражал захваченные фрагменты без ручного клика по «Обновить».
    if not progress.get("is_complete", False):
        st_autorefresh(interval=2000, key="enroll-progress-refresh", limit=None)

    current_target = None
    for s in schedule:
        emo = s["emotion"]
        p = progress["progress"].get(emo, {})
        if p.get("current", 0) < p.get("target", s["count"]):
            current_target = emo
            break

    if current_target:
        st.markdown(
            f"### Сейчас сделай: **{EMOTION_LABELS[current_target]}** "
            f"({progress['progress'][current_target]['current']}/"
            f"{progress['progress'][current_target]['target']})"
        )
    else:
        st.success("✅ Все эмоции собраны! Переходите к шагу 3.")

    cols = st.columns(len(schedule))
    for col, s in zip(cols, schedule):
        emo = s["emotion"]
        p = progress["progress"].get(emo, {})
        cur = p.get("current", 0)
        tgt = p.get("target", s["count"])
        emoji = EMOTION_EMOJI.get(emo, "❓")
        is_active = emo == current_target
        col.metric(
            label=f"{emoji} {emo}{' ◀' if is_active else ''}",
            value=f"{cur}/{tgt}",
        )

    ctx = webrtc_streamer(
        key="enroll-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EnrollProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.session_id = session_id
        ctx.video_processor.schedule = schedule_dict
        ctx.video_processor.current_target = current_target
        with ctx.video_processor.lock:
            for emo, p in progress["progress"].items():
                ctx.video_processor.uploaded_counts[emo] = p.get("current", 0)
        if ctx.video_processor.error_message:
            st.warning(ctx.video_processor.error_message)

    col1, col2, col3 = st.columns([1, 1, 1])
    if col1.button("🔄 Обновить прогресс"):
        st.rerun()
    if col2.button("⬅ Назад"):
        _reset_state()
        st.rerun()
    if col3.button(
        "🎓 Обучить НПБК",
        type="primary",
        disabled=not progress.get("is_complete", False),
    ):
        st.session_state.enroll_step = 3
        st.rerun()


def _step_finalize() -> None:
    st.subheader("Шаг 3: обучение НПБК")
    session_id = st.session_state.session_id

    if "enroll_result" not in st.session_state:
        with st.spinner("Обучение НПБК и шифрование ключа… (может занять до минуты)"):
            try:
                r = httpx.post(
                    f"{API_URL}/api/enroll/finalize",
                    json={"session_id": session_id},
                    timeout=180,
                )
                r.raise_for_status()
                st.session_state.enroll_result = r.json()
            except Exception as e:
                st.error(f"Ошибка финализации: {e}")
                if st.button("⬅ Назад к записи"):
                    st.session_state.enroll_step = 2
                    st.rerun()
                return

    result = st.session_state.enroll_result
    st.success(f"✅ Пользователь **{result['display_name']}** зарегистрирован!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Векторов", result["n_vectors"])
    c2.metric("Стабильность", f"{result['mean_stability']:.3f}")
    c3.metric("Длина кода", result["code_length"])

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)
    if result.get("failed_fragments"):
        st.info(f"Битых фрагментов: {result['failed_fragments']}")

    st.markdown("### SSH публичный ключ")
    st.caption("Добавьте эту строку в `~/.ssh/authorized_keys` на целевом сервере.")
    st.code(result["public_key"], language=None)

    try:
        cfg_r = httpx.get(
            f"{API_URL}/api/users/{result['uid']}/config",
            timeout=10,
            follow_redirects=True,
        )
        cfg_r.raise_for_status()
        config_bytes = cfg_r.content
        filename = f"{result['display_name']}-{result['uid'][:8]}.json"
        st.download_button(
            "💾 Скачать client config.json",
            data=config_bytes,
            file_name=filename,
            mime="application/json",
            type="primary",
        )
    except Exception as e:
        st.warning(f"Не удалось получить config.json: {e}")

    if st.button("🔄 Зарегистрировать ещё одного"):
        _reset_state()
        st.rerun()


def render() -> None:
    st.header("➕ Регистрация нового пользователя")
    step = st.session_state.get("enroll_step", 1)

    pcols = st.columns(3)
    for i, label in enumerate(("1. Настройки", "2. Запись", "3. Обучение"), start=1):
        with pcols[i - 1]:
            mark = "✅" if i < step else ("▶" if i == step else "·")
            st.markdown(f"**{mark} {label}**")
    st.divider()

    if step == 1:
        _step_settings()
    elif step == 2:
        _step_recording()
    elif step == 3:
        _step_finalize()
