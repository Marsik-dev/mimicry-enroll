from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EnrolledUser(Base):
    __tablename__ = "enrolled_users"

    uid: Mapped[str] = mapped_column(String(36), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    main_emotion: Mapped[str] = mapped_column(String(16), nullable=False)
    key_type: Mapped[str] = mapped_column(String(16), default="ed25519")
    reference_container: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encrypted_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_salt: Mapped[bytes] = mapped_column(LargeBinary(16), nullable=False)
    public_key: Mapped[str] = mapped_column(Text, nullable=False)
    enrolled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    n_vectors: Mapped[int] = mapped_column(Integer, default=0)
    mean_stability: Mapped[float] = mapped_column(Float, default=0.0)
    code_length: Mapped[int] = mapped_column(Integer, default=0)
