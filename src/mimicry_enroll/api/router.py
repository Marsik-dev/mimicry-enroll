"""FastAPI REST API для CLI клиентов."""
from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse

from mimicry_enroll.crypto import encrypt_key
from mimicry_enroll.db.models import EnrolledUser
from mimicry_enroll.db.session import get_session
from mimicry_enroll.enrollor import enroll

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/users/{uid}/enroll")
async def enroll_user(uid: str, videos: list[UploadFile] = File(...)):
    """Принять видео файлы, провести enrollment, вернуть публичный SSH ключ."""
    if not uid or len(uid) > 64:
        raise HTTPException(400, "uid must be 1-64 chars")

    with tempfile.TemporaryDirectory(prefix=f"enroll_{uid}_") as tmpdir:
        video_paths = []
        for vf in videos:
            dest = Path(tmpdir) / vf.filename
            dest.write_bytes(await vf.read())
            video_paths.append(dest)

        try:
            result = enroll(video_paths, uid)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            log.exception("Enrollment failed for %s", uid)
            raise HTTPException(500, f"Enrollment error: {e}")

    salt = os.urandom(16)
    encrypted_key = encrypt_key(result.reference_code, result.private_key_pem, salt)

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
        raise HTTPException(500, f"DB error: {e}")
    finally:
        session.close()

    return {
        "uid": uid,
        "public_key": result.public_key_text,
        "n_vectors": result.n_vectors,
        "mean_stability": round(result.mean_stability, 4),
        "code_length": result.code_length,
        "warnings": result.warnings,
    }


@router.get("/users/{uid}/config")
def get_config(uid: str):
    """Скачать client config JSON (reference_container + encrypted_key)."""
    session = get_session()
    try:
        user = session.get(EnrolledUser, uid)
        if not user:
            raise HTTPException(404, f"User '{uid}' not found")
        config = {
            "uid": uid,
            "reference_container": base64.b64encode(user.reference_container).decode(),
            "encrypted_key": base64.b64encode(user.encrypted_key).decode(),
            "key_salt": base64.b64encode(user.key_salt).decode(),
        }
    finally:
        session.close()
    return JSONResponse(content=config, headers={
        "Content-Disposition": f'attachment; filename="{uid}.json"'
    })


@router.get("/users/{uid}/public-key", response_class=PlainTextResponse)
def get_public_key(uid: str):
    """Вернуть SSH публичный ключ (для добавления в authorized_keys)."""
    session = get_session()
    try:
        user = session.get(EnrolledUser, uid)
        if not user:
            raise HTTPException(404, f"User '{uid}' not found")
        return user.public_key
    finally:
        session.close()


@router.delete("/users/{uid}")
def delete_user(uid: str):
    session = get_session()
    try:
        user = session.get(EnrolledUser, uid)
        if not user:
            raise HTTPException(404, f"User '{uid}' not found")
        session.delete(user)
        session.commit()
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"DB error: {e}")
    finally:
        session.close()
    return {"deleted": uid}


@router.get("/users")
def list_users():
    from mimicry_enroll.db.models import EnrolledUser as EU
    from sqlalchemy import select
    session = get_session()
    try:
        rows = session.execute(select(EU)).scalars().all()
        return [
            {
                "uid": u.uid,
                "n_vectors": u.n_vectors,
                "mean_stability": round(u.mean_stability, 4),
                "code_length": u.code_length,
                "enrolled_at": u.enrolled_at.isoformat() if u.enrolled_at else None,
            }
            for u in rows
        ]
    finally:
        session.close()
