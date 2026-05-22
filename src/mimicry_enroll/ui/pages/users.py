"""Страница: список зарегистрированных пользователей."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any

import streamlit as st
from sqlalchemy import select

from mimicry_enroll.db.models import EnrolledUser
from mimicry_enroll.db.session import get_session

EMOTION_EMOJI = {
    "happy": "😊",
    "angry": "😠",
    "surprise": "😲",
    "sad": "😢",
}


def _fmt_date(dt: datetime | None) -> str:
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


def _row_to_dict(user: EnrolledUser) -> dict[str, Any]:
    """Слепок ORM-объекта в простой dict, чтобы безопасно работать после db.close()."""
    return {
        "uid": user.uid,
        "display_name": user.display_name,
        "main_emotion": user.main_emotion,
        "key_type": user.key_type,
        "enrolled_at": user.enrolled_at,
        "n_vectors": user.n_vectors,
        "mean_stability": user.mean_stability,
        "code_length": user.code_length,
        "public_key": user.public_key,
        "reference_container": user.reference_container,
        "encrypted_key": user.encrypted_key,
        "key_salt": user.key_salt,
    }


def _config_bytes(u: dict[str, Any]) -> bytes:
    payload = {
        "uid": u["uid"],
        "display_name": u["display_name"],
        "main_emotion": u["main_emotion"],
        "reference_container": base64.b64encode(u["reference_container"]).decode(),
        "encrypted_key": base64.b64encode(u["encrypted_key"]).decode(),
        "key_salt": base64.b64encode(u["key_salt"]).decode(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode()


def _load_users() -> list[dict[str, Any]]:
    db = get_session()
    try:
        rows = db.execute(
            select(EnrolledUser).order_by(EnrolledUser.enrolled_at.desc())
        ).scalars().all()
        return [_row_to_dict(u) for u in rows]
    finally:
        db.close()


def _delete_user(uid: str) -> None:
    db = get_session()
    try:
        user = db.get(EnrolledUser, uid)
        if user is not None:
            db.delete(user)
            db.commit()
    finally:
        db.close()


def render() -> None:
    st.header("👥 Зарегистрированные пользователи")

    users = _load_users()

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

    for u in users:
        uid = u["uid"]
        cols = st.columns([2.5, 2, 1.5, 1.2, 1.8, 1.5, 1])
        cols[0].write(u["display_name"])
        cols[1].code(uid[:8] + "…", language=None)
        emoji = EMOTION_EMOJI.get(u["main_emotion"], "❓")
        cols[2].write(f"{emoji} {u['main_emotion']}")
        cols[3].write(u["key_type"])
        cols[4].write(_fmt_date(u["enrolled_at"]))

        filename = f"{u['display_name']}-{uid[:8]}.json"
        cols[5].download_button(
            "⬇ Скачать",
            data=_config_bytes(u),
            file_name=filename,
            mime="application/json",
            key=f"dl_{uid}",
            use_container_width=True,
        )

        if cols[6].button("🗑", key=f"del_{uid}", help="Удалить пользователя"):
            _delete_user(uid)
            st.rerun()

    with st.expander("Подробности (векторы, стабильность, public keys)"):
        for u in users:
            st.markdown(f"**{u['display_name']}** — `{u['uid']}`")
            st.text(
                f"  Векторов: {u['n_vectors']}  "
                f"Стабильность: {u['mean_stability']:.3f}  "
                f"Длина кода: {u['code_length']}"
            )
            st.code(u["public_key"], language=None)
