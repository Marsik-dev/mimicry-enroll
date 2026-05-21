"""Страница: список зарегистрированных пользователей."""
from __future__ import annotations

import streamlit as st

from mimicry_enroll.db.models import EnrolledUser
from mimicry_enroll.db.session import get_session


def render():
    st.header("Зарегистрированные пользователи")

    session = get_session()
    try:
        from sqlalchemy import select
        users = session.execute(select(EnrolledUser)).scalars().all()
    finally:
        session.close()

    if not users:
        st.info("Нет зарегистрированных пользователей. Перейдите на вкладку «Регистрация».")
        return

    for u in users:
        with st.expander(f"👤 {u.uid}  —  {u.n_vectors} векторов, stability={u.mean_stability:.3f}"):
            col1, col2 = st.columns(2)
            col1.metric("Векторов", u.n_vectors)
            col1.metric("Стабильность НПБК", f"{u.mean_stability:.3f}")
            col2.metric("Длина кода", u.code_length)
            col2.metric("Дата", u.enrolled_at.strftime("%Y-%m-%d %H:%M") if u.enrolled_at else "—")

            st.code(u.public_key, language=None)

            col_dl, col_del = st.columns(2)
            if col_dl.button("Скачать config.json", key=f"dl_{u.uid}"):
                import base64
                import json
                config = {
                    "uid": u.uid,
                    "reference_container": base64.b64encode(u.reference_container).decode(),
                    "encrypted_key": base64.b64encode(u.encrypted_key).decode(),
                    "key_salt": base64.b64encode(u.key_salt).decode(),
                }
                st.download_button(
                    "💾 Сохранить",
                    data=json.dumps(config, indent=2),
                    file_name=f"{u.uid}.json",
                    mime="application/json",
                    key=f"save_{u.uid}",
                )

            if col_del.button("🗑 Удалить", key=f"del_{u.uid}", type="secondary"):
                session2 = get_session()
                try:
                    user2 = session2.get(EnrolledUser, u.uid)
                    if user2:
                        session2.delete(user2)
                        session2.commit()
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                finally:
                    session2.close()
