"""Страница: список зарегистрированных пользователей."""
from __future__ import annotations

import base64
import json
from datetime import datetime

import streamlit as st
from sqlalchemy import select

from mimicry_enroll.db.models import EnrolledUser
from mimicry_enroll.db.session import get_session

EMOTION_EMOJI = {
    "happy": "😊",
    "angry": "😠",
    "surprise": "😲",
    "sad": "😢",
    "disgust": "🤢",
}


def _fmt_date(dt: datetime | None) -> str:
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


def _config_bytes(user: EnrolledUser) -> bytes:
    payload = {
        "uid": user.uid,
        "display_name": user.display_name,
        "main_emotion": user.main_emotion,
        "reference_container": base64.b64encode(user.reference_container).decode(),
        "encrypted_key": base64.b64encode(user.encrypted_key).decode(),
        "key_salt": base64.b64encode(user.key_salt).decode(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode()


def render():
    st.header("👥 Зарегистрированные пользователи")

    db = get_session()
    try:
        users = db.execute(
            select(EnrolledUser).order_by(EnrolledUser.enrolled_at.desc())
        ).scalars().all()
    finally:
        db.close()

    if not users:
        st.info("Пока никого. Перейдите в **Регистрация** чтобы добавить.")
        return

    st.caption(f"Всего: {len(users)}")

    header_cols = st.columns([2.5, 2, 1.5, 1.2, 1.8, 1.5, 1])
    for col, label in zip(
        header_cols,
        ("Имя", "UID", "Эмоция", "Ключ", "Зарегистрирован", "Config", ""),
    ):
        col.markdown(f"**{label}**")
    st.divider()

    for user in users:
        cols = st.columns([2.5, 2, 1.5, 1.2, 1.8, 1.5, 1])
        cols[0].write(user.display_name)
        cols[1].code(user.uid[:8] + "…", language=None)
        emoji = EMOTION_EMOJI.get(user.main_emotion, "❓")
        cols[2].write(f"{emoji} {user.main_emotion}")
        cols[3].write(user.key_type)
        cols[4].write(_fmt_date(user.enrolled_at))

        filename = f"{user.display_name}-{user.uid[:8]}.json"
        cols[5].download_button(
            "⬇ Скачать",
            data=_config_bytes(user),
            file_name=filename,
            mime="application/json",
            key=f"dl_{user.uid}",
            use_container_width=True,
        )

        if cols[6].button("🗑", key=f"del_{user.uid}", help="Удалить пользователя"):
            db = get_session()
            try:
                db.delete(db.get(EnrolledUser, user.uid))
                db.commit()
            finally:
                db.close()
            st.rerun()

    with st.expander("Подробности (векторы, стабильность, public keys)"):
        for user in users:
            st.markdown(f"**{user.display_name}** — `{user.uid}`")
            st.text(
                f"  Векторов: {user.n_vectors}  "
                f"Стабильность: {user.mean_stability:.3f}  "
                f"Длина кода: {user.code_length}"
            )
            st.code(user.public_key, language=None)
