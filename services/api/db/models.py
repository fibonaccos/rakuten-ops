from enum import StrEnum
from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, func, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        String(50),
        nullable=False,
        default=UserRole.USER
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    inferences = relationship(
        "Inference",
        back_populates="user"
    )


class Inference(Base):
    __tablename__ = "inference"

    inference_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer(),
        ForeignKey("users.user_id"),
        nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(100), nullable=False)
    batch: Mapped[bool] = mapped_column(Boolean(), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(10), nullable=False)
    designation: Mapped[str] = mapped_column(Text(), nullable=False)
    description: Mapped[str] = mapped_column(Text(), nullable=True)
    predicted_category: Mapped[str] = mapped_column(String(50), nullable=False)
    labeled_category: Mapped[str] = mapped_column(String(50), nullable=True)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False)
    queried_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    user = relationship(
        "User",
        back_populates="inferences"
    )
