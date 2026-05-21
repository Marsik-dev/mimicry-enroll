"""FastAPI REST API: real-time session-based enrollment + управление пользователями."""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from mimicry_enroll.crypto import encrypt_key
from mimicry_enroll.db.models import EnrolledUser
from mimicry_enroll.db.session import get_session
from mimicry_enroll.enrollor import enroll, extract_vectors_from_videos

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

SESSION_TTL = 600  # 10 минут
SUPPORTED_EMOTIONS = ("happy", "angry", "surprise", "sad", "disgust")
KeyType = Literal["ed25519", "rsa", "imported"]

# Расписание сбора: основная эмоция × 8, остальные × 2 (выбираем 3 случайных из 4)
SCHEDULE_MAIN_COUNT = 8
SCHEDULE_OTHER_COUNT = 2
SCHEDULE_N_OTHER_EMOTIONS = 3

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="enroll-pipeline")


@dataclass
class Fragment:
    emotion: str
    video_path: Path
    future: Future | None = None  # future с np.ndarray | None
    vector: np.ndarray | None = None
    error: str | None = None


@dataclass
class EnrollSession:
    session_id: str
    display_name: str
    main_emotion: str
    key_type: KeyType
    imported_pem: bytes | None
    schedule: dict[str, int]  # emotion → target_count
    tmp_dir: Path
    fragments: list[Fragment] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def counts(self) -> dict[str, int]:
        c = {e: 0 for e in self.schedule}
        for fr in self.fragments:
            c[fr.emotion] = c.get(fr.emotion, 0) + 1
        return c

    def progress(self) -> dict[str, dict]:
        c = self.counts()
        return {
            e: {"current": c.get(e, 0), "target": self.schedule[e]}
            for e in self.schedule
        }

    def is_complete(self) -> bool:
        c = self.counts()
        return all(c.get(e, 0) >= n for e, n in self.schedule.items())

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


SESSIONS: dict[str, EnrollSession] = {}


def _cleanup_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.created_at > SESSION_TTL]
    for sid in expired:
        s = SESSIONS.pop(sid, None)
        if s:
            s.cleanup()
            log.info("[%s] Сессия очищена по TTL", sid[:8])


@router.get("/health")
def health():
    return {"status": "ok"}


# ────────────────────────────────────────────────────────────────────────────
# Session-based real-time enrollment
# ────────────────────────────────────────────────────────────────────────────

class StartReq(BaseModel):
    display_name: str
    main_emotion: str
    key_type: KeyType = "ed25519"
    imported_pem_b64: str | None = None


@router.post("/enroll/start")
def enroll_start(req: StartReq):
    _cleanup_expired()

    if not req.display_name.strip() or len(req.display_name) > 64:
        raise HTTPException(400, "display_name должно быть 1-64 символа")
    if req.main_emotion not in SUPPORTED_EMOTIONS:
        raise HTTPException(400, f"main_emotion должно быть одним из {SUPPORTED_EMOTIONS}")

    imported_pem = None
    if req.key_type == "imported":
        if not req.imported_pem_b64:
            raise HTTPException(400, "imported_pem_b64 обязателен при key_type='imported'")
        try:
            imported_pem = base64.b64decode(req.imported_pem_b64)
        except Exception as e:
            raise HTTPException(400, f"Невалидный imported_pem_b64: {e}")

    others = [e for e in SUPPORTED_EMOTIONS if e != req.main_emotion]
    rng = np.random.default_rng()
    others_chosen = list(rng.choice(others, size=SCHEDULE_N_OTHER_EMOTIONS, replace=False))

    schedule = {req.main_emotion: SCHEDULE_MAIN_COUNT}
    for e in others_chosen:
        schedule[str(e)] = SCHEDULE_OTHER_COUNT

    session_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"enroll_{session_id[:8]}_"))
    sess = EnrollSession(
        session_id=session_id,
        display_name=req.display_name.strip(),
        main_emotion=req.main_emotion,
        key_type=req.key_type,
        imported_pem=imported_pem,
        schedule=schedule,
        tmp_dir=tmp_dir,
    )
    SESSIONS[session_id] = sess
    log.info("[%s] Новая сессия: %s, main=%s, schedule=%s",
             session_id[:8], sess.display_name, sess.main_emotion, schedule)

    return {
        "session_id": session_id,
        "schedule": [{"emotion": e, "count": n} for e, n in schedule.items()],
    }


def _process_fragment(fragment: Fragment) -> None:
    """Извлечь вектор из видеофрагмента (выполняется в треде executor)."""
    try:
        vectors = extract_vectors_from_videos([fragment.video_path])
        if vectors:
            fragment.vector = vectors[0]
        else:
            fragment.error = "пустой вектор после Pipeline"
    except Exception as e:
        fragment.error = str(e)


@router.post("/enroll/upload-fragment")
async def enroll_upload_fragment(
    session_id: str = Form(...),
    emotion: str = Form(...),
    video: UploadFile = File(...),
):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found or expired")
    if emotion not in sess.schedule:
        raise HTTPException(400, f"emotion '{emotion}' не в расписании сессии")

    counts = sess.counts()
    if counts.get(emotion, 0) >= sess.schedule[emotion]:
        return JSONResponse({"progress": sess.progress(), "ignored": True})

    fragment_path = sess.tmp_dir / f"{emotion}_{len(sess.fragments):03d}_{uuid.uuid4().hex[:6]}.mp4"
    fragment_path.write_bytes(await video.read())

    fragment = Fragment(emotion=emotion, video_path=fragment_path)
    sess.fragments.append(fragment)

    fragment.future = _EXECUTOR.submit(_process_fragment, fragment)

    return {"progress": sess.progress()}


@router.get("/enroll/progress")
def enroll_progress(session_id: str):
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found or expired")

    processed = sum(1 for fr in sess.fragments if fr.vector is not None or fr.error is not None)
    return {
        "progress": sess.progress(),
        "fragments_total": len(sess.fragments),
        "fragments_processed": processed,
        "is_complete": sess.is_complete(),
    }


class FinalizeReq(BaseModel):
    session_id: str


@router.post("/enroll/finalize")
def enroll_finalize(req: FinalizeReq, bg: BackgroundTasks):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found or expired")

    # Дождаться обработки всех фрагментов
    for fr in sess.fragments:
        if fr.future is not None:
            try:
                fr.future.result(timeout=120)
            except Exception as e:
                fr.error = str(e)

    main_vecs: list[np.ndarray] = []
    other_vecs: list[np.ndarray] = []
    failed = 0
    for fr in sess.fragments:
        if fr.vector is None:
            failed += 1
            continue
        if fr.emotion == sess.main_emotion:
            main_vecs.append(fr.vector)
        else:
            other_vecs.append(fr.vector)

    if len(main_vecs) < 3:
        raise HTTPException(
            422,
            f"Не хватает векторов основной эмоции (есть {len(main_vecs)}, нужно ≥3). "
            f"Битых фрагментов: {failed}.",
        )

    try:
        result = enroll(
            main_emotion_vectors=main_vecs,
            other_emotion_vectors=other_vecs,
            main_emotion=sess.main_emotion,
            key_type=sess.key_type,
            imported_pem=sess.imported_pem,
        )
    except Exception as e:
        log.exception("[%s] Enrollment failed", req.session_id[:8])
        raise HTTPException(500, f"Enrollment error: {e}")

    salt = os.urandom(16)
    encrypted_key = encrypt_key(result.reference_code, result.private_key_pem, salt)

    uid = str(uuid.uuid4())
    db = get_session()
    try:
        user = EnrolledUser(
            uid=uid,
            display_name=sess.display_name,
            main_emotion=sess.main_emotion,
            key_type=sess.key_type,
            reference_container=result.container_bytes,
            encrypted_key=encrypted_key,
            key_salt=salt,
            public_key=result.public_key_text,
            n_vectors=result.n_vectors,
            mean_stability=result.mean_stability,
            code_length=result.code_length,
        )
        db.add(user)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"DB error: {e}")
    finally:
        db.close()

    bg.add_task(_finalize_cleanup, req.session_id)

    return {
        "uid": uid,
        "display_name": sess.display_name,
        "main_emotion": sess.main_emotion,
        "key_type": sess.key_type,
        "public_key": result.public_key_text,
        "n_vectors": result.n_vectors,
        "mean_stability": round(result.mean_stability, 4),
        "code_length": result.code_length,
        "warnings": result.warnings,
        "failed_fragments": failed,
    }


def _finalize_cleanup(session_id: str) -> None:
    sess = SESSIONS.pop(session_id, None)
    if sess:
        sess.cleanup()


# ────────────────────────────────────────────────────────────────────────────
# User management
# ────────────────────────────────────────────────────────────────────────────


@router.get("/users/{uid}/config")
def get_config(uid: str):
    db = get_session()
    try:
        user = db.get(EnrolledUser, uid)
        if not user:
            raise HTTPException(404, f"User '{uid}' not found")
        config = {
            "uid": uid,
            "display_name": user.display_name,
            "main_emotion": user.main_emotion,
            "reference_container": base64.b64encode(user.reference_container).decode(),
            "encrypted_key": base64.b64encode(user.encrypted_key).decode(),
            "key_salt": base64.b64encode(user.key_salt).decode(),
        }
    finally:
        db.close()
    filename = f"{user.display_name}-{uid[:8]}.json"
    return JSONResponse(
        content=config,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/users/{uid}/public-key", response_class=PlainTextResponse)
def get_public_key(uid: str):
    db = get_session()
    try:
        user = db.get(EnrolledUser, uid)
        if not user:
            raise HTTPException(404, f"User '{uid}' not found")
        return user.public_key
    finally:
        db.close()


@router.delete("/users/{uid}")
def delete_user(uid: str):
    db = get_session()
    try:
        user = db.get(EnrolledUser, uid)
        if not user:
            raise HTTPException(404, f"User '{uid}' not found")
        db.delete(user)
        db.commit()
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"DB error: {e}")
    finally:
        db.close()
    return {"deleted": uid}


@router.get("/users")
def list_users():
    from sqlalchemy import select
    db = get_session()
    try:
        rows = db.execute(select(EnrolledUser).order_by(EnrolledUser.enrolled_at.desc())).scalars().all()
        return [
            {
                "uid": u.uid,
                "display_name": u.display_name,
                "main_emotion": u.main_emotion,
                "key_type": u.key_type,
                "n_vectors": u.n_vectors,
                "mean_stability": round(u.mean_stability, 4),
                "code_length": u.code_length,
                "enrolled_at": u.enrolled_at.isoformat() if u.enrolled_at else None,
            }
            for u in rows
        ]
    finally:
        db.close()
