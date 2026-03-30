import os

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL не задан в переменных окружения. "
        "Установите переменную для подключения к PostgreSQL."
    )
SCHEMA = "public"


def get_engine(schema: str) -> AsyncEngine:
    """Создаёт и возвращает асинхронный движок SQLAlchemy с указанным схемой.

    Устанавливает параметр search_path в соединении, чтобы все запросы выполнялись
    в заданной схеме PostgreSQL.
    """
    return create_async_engine(
        DATABASE_URL,
        connect_args={"server_settings": {"search_path": schema}},
        pool_pre_ping=True,
        pool_recycle=1800,
    )


if SCHEMA is None or SCHEMA == "":
    engine = get_engine("public")
else:
    engine = get_engine(SCHEMA)
async_session = async_sessionmaker(engine)


class Base(AsyncAttrs, DeclarativeBase):
    """Базовый класс для всех моделей, поддерживающий асинхронные атрибуты."""

    pass


class User(Base):
    """Модель пользователя Discord."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    name = Column(String(50), nullable=False)
    context = Column(JSONB)
    datetime_insert = Column(DateTime, default=func.now())


class ChannelMessage(Base):
    """Модель сообщения в канале Discord."""

    __tablename__ = "channel_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(BigInteger, nullable=False)
    message_id = Column(BigInteger, nullable=False)
    author = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now())


class Birthday(Base):
    """Модель дня рождения пользователя."""

    __tablename__ = "birthday"
    user_id = Column(BigInteger, primary_key=True, nullable=False)
    display_name = Column(String(50), nullable=False)
    name = Column(String(50), nullable=False)
    birthday = Column(Date, nullable=False)
    datetime_insert = Column(DateTime, default=func.now())


class UserMessageStats(Base):
    """Модель статистики сообщений пользователя."""

    __tablename__ = "user_message_stats"

    user_id = Column(BigInteger, primary_key=True, nullable=False)
    guild_id = Column(BigInteger, primary_key=True, nullable=False)
    name = Column(String(50), nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_guild_message_count", "guild_id", "message_count"),
        Index("idx_user_activity", "last_updated"),
    )


class YouTubeChannel(Base):
    """Модель отслеживаемого YouTube канала."""

    __tablename__ = "youtube_channels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(String(50), nullable=False)
    guild_id = Column(BigInteger)
    discord_channel_id = Column(BigInteger, nullable=False)
    name = Column(String(100), nullable=False)
    last_checked = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)


class YouTubeVideo(Base):
    """Модель видео с YouTube."""

    __tablename__ = "youtube_videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(50), nullable=False)
    guild_id = Column(BigInteger, nullable=False)
    channel_id = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    published_at = Column(DateTime, nullable=False)
    is_live = Column(Boolean, default=False, nullable=False)

    __table_args__ = (UniqueConstraint("video_id", "guild_id", name="uq_youtube_video_per_guild"),)


class Holiday(Base):
    """Модель праздника."""

    __tablename__ = "holidays"

    day = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)

    __table_args__ = (PrimaryKeyConstraint("day", "month", name="pk_holidays"),)


async def init_models() -> None:
    """Создает таблицы в базе данных, если они не существуют."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
