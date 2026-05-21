"""Точка входа: запускает Streamlit UI + FastAPI в отдельном потоке."""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)


def _run_api():
    import uvicorn
    from fastapi import FastAPI
    from mimicry_enroll.api.router import router
    from mimicry_enroll.config import settings
    from mimicry_enroll.db.session import init_db

    app = FastAPI(title="mimicry-enroll API", version="0.1.0")
    app.include_router(router)
    init_db()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from mimicry_enroll.config import settings

    # FastAPI в фоновом потоке
    api_thread = threading.Thread(target=_run_api, daemon=True, name="api-server")
    api_thread.start()
    log.info("API сервер запущен на порту %d", settings.api_port)

    # Streamlit в основном процессе
    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(settings.streamlit_port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ]
    log.info("Streamlit UI запускается на порту %d", settings.streamlit_port)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
