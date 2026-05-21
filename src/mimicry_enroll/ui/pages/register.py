"""Страница: регистрация нового пользователя."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from mimicry_enroll.config import settings
from mimicry_enroll.crypto import encrypt_key
from mimicry_enroll.db.models import EnrolledUser
from mimicry_enroll.db.session import get_session
from mimicry_enroll.enrollor import enroll


def render():
    st.header("Регистрация нового пользователя")

    uid = st.text_input("UID пользователя", placeholder="alice", max_chars=64)

    st.subheader("Загрузить видео с лицом")
    st.caption(
        "Запишите несколько коротких видео (3–10 сек) с лицом при разном освещении и углах. "
        "Чем больше видео — тем лучше стабильность НПБК."
    )

    uploaded = st.file_uploader(
        "Видео файлы (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )

    if not uid:
        st.warning("Введите UID пользователя.")
        return

    if not uploaded:
        st.info("Загрузите хотя бы один видео файл.")
        return

    if st.button("🚀 Запустить enrollment", type="primary"):
        progress = st.progress(0, text="Сохраняю видео...")
        status = st.empty()

        with tempfile.TemporaryDirectory(prefix=f"enroll_{uid}_") as tmpdir:
            video_paths = []
            for i, f in enumerate(uploaded):
                dest = Path(tmpdir) / f.name
                dest.write_bytes(f.read())
                video_paths.append(dest)
                progress.progress((i + 1) / (len(uploaded) + 3), text=f"Сохранено: {f.name}")

            status.info("Извлекаю векторы признаков через Pipeline...")
            progress.progress(len(uploaded) / (len(uploaded) + 3), text="Pipeline...")

            try:
                result = enroll(video_paths, uid)
            except ValueError as e:
                st.error(f"Ошибка: {e}")
                return
            except Exception as e:
                st.error(f"Ошибка enrollment: {e}")
                st.exception(e)
                return

        progress.progress(0.85, text="Шифрование SSH ключа...")
        salt = os.urandom(16)
        encrypted_key = encrypt_key(result.reference_code, result.private_key_pem, salt)

        status.info("Сохраняю в базу данных...")
        session = get_session()
        try:
            existing = session.get(EnrolledUser, uid)
            user = existing or EnrolledUser(uid=uid)
            user.reference_container = result.container_bytes
            user.encrypted_key = encrypted_key
            user.key_salt = salt
            user.public_key = result.public_key_text
            user.n_vectors = result.n_vectors
            user.mean_stability = result.mean_stability
            user.code_length = result.code_length
            if not existing:
                session.add(user)
            session.commit()
        except Exception as e:
            session.rollback()
            st.error(f"Ошибка БД: {e}")
            return
        finally:
            session.close()

        progress.progress(1.0, text="Готово!")
        status.success(f"✅ Пользователь **{uid}** успешно зарегистрирован!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Векторов", result.n_vectors)
        col2.metric("Стабильность НПБК", f"{result.mean_stability:.3f}")
        col3.metric("Длина кода", result.code_length)

        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        st.subheader("SSH публичный ключ")
        st.caption("Добавьте эту строку в ~/.ssh/authorized_keys на целевом сервере.")
        st.code(result.public_key_text, language=None)

        import base64, json
        config = {
            "uid": uid,
            "reference_container": base64.b64encode(result.container_bytes).decode(),
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "key_salt": base64.b64encode(salt).decode(),
        }
        st.download_button(
            "💾 Скачать client config.json",
            data=json.dumps(config, indent=2),
            file_name=f"{uid}.json",
            mime="application/json",
        )
