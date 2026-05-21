from __future__ import annotations

import os
from pathlib import Path


class Settings:
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://mimicry:mimicry123@localhost:5432/mimicry_enroll"
    )
    streamlit_port: int = int(os.environ.get("STREAMLIT_PORT", "8501"))
    api_port: int = int(os.environ.get("API_PORT", "8000"))
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    min_vectors: int = int(os.environ.get("MIN_VECTORS", "11"))
    app_title: str = "НПБК Enrollment Server"


settings = Settings()
