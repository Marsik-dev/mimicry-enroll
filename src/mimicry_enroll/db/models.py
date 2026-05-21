from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EnrolledUser(Base):
    __tablename__ = "enrolled_users"

    uid: Mapped[str] = mapped_column(String(64), primary_key=True)
    # NBKContainer serialized via .to_bytes()
    reference_container: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # SSH private key encrypted with НПБК reference_code (Grasshopper CTR)
    encrypted_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # KDF salt for Grasshopper
    key_salt: Mapped[bytes] = mapped_column(LargeBinary(16), nullable=False)
    # SSH public key text (OpenSSH format, line to add to authorized_keys)
    public_key: Mapped[str] = mapped_column(Text, nullable=False)
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    n_vectors: Mapped[int] = mapped_column(Integer, default=0)
    mean_stability: Mapped[float] = mapped_column(Float, default=0.0)
    code_length: Mapped[int] = mapped_column(Integer, default=0)
