"""Точка входа: запускает Streamlit UI + FastAPI в отдельном потоке.

Если в /app/certs/ лежат cert.pem и key.pem — обе службы стартуют с HTTPS.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

CERT_DIR = Path(os.environ.get("SSL_CERT_DIR", "/app/certs"))
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"


def _ssl_enabled() -> bool:
    return CERT_FILE.exists() and KEY_FILE.exists()


def _run_api():
    import uvicorn
    from fastapi import FastAPI
    from mimicry_enroll.api.router import router
    from mimicry_enroll.config import settings
    from mimicry_enroll.db.session import init_db

    app = FastAPI(title="mimicry-enroll API", version="0.1.0")
    app.include_router(router)
    init_db()

    # FastAPI остаётся на HTTP — это внутренний вызов из того же контейнера.
    # SSL включаем только на Streamlit (внешний доступ с телефона/LAN).
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from mimicry_enroll.config import settings

    if _ssl_enabled():
        log.info("SSL включён: %s + %s", CERT_FILE, KEY_FILE)
    else:
        log.info("SSL не настроен (нет cert.pem/key.pem в %s) — работаем по HTTP", CERT_DIR)

    api_thread = threading.Thread(target=_run_api, daemon=True, name="api-server")
    api_thread.start()
    log.info("API сервер запущен на порту %d", settings.api_port)

    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(settings.streamlit_port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    if _ssl_enabled():
        cmd += [
            "--server.sslCertFile", str(CERT_FILE),
            "--server.sslKeyFile", str(KEY_FILE),
        ]
    log.info("Streamlit UI запускается на порту %d", settings.streamlit_port)
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
