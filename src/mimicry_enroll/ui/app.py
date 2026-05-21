"""Streamlit приложение — enrollment сервер."""
from __future__ import annotations

import streamlit as st

from mimicry_enroll.config import settings
from mimicry_enroll.db.session import init_db

st.set_page_config(
    page_title=settings.app_title,
    page_icon="🔐",
    layout="wide",
)


@st.cache_resource
def _init():
    init_db()


_init()

tab_users, tab_register = st.tabs(["👥 Пользователи", "➕ Регистрация"])

with tab_users:
    from mimicry_enroll.ui.pages.users import render as render_users
    render_users()

with tab_register:
    from mimicry_enroll.ui.pages.register import render as render_register
    render_register()
