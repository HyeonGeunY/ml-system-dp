from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from src.db.database import Base

# Projects 테이블
class Project(Base):
    __tablename__ = "projects"

    project_id = Column(
        String(255),
        primary_key=True,
        comment="기본키",
    )
    project_name = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="프로젝트명",
    )
    description = Column(
        Text,
        nullable=True,
        comment="설명",
    )

    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )


# models 테이블
class Model(Base):
    __tablename__ = "models"

    model_id = Column(
        String(255),
        primary_key=True,
        comment="기본키",
    )
    project_id = Column(
        String(255),
        ForeignKey("projects.project_id"),
        nullable=False,
        comment="외부키",
    )
    model_name = Column(
        String(255),
        nullable=False,
        comment="モデル名",
    )
