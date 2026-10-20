import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func

from app.core.config import get_database_settings

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL не задан в переменных окружения. "
        "Установите переменную для подключения к PostgreSQL."
    )

SCHEMA: str = get_database_settings().schema

engine = create_async_engine(
    DATABASE_URL,
    connect_args={"server_settings": {"search_path": SCHEMA if SCHEMA else "public"}},
    pool_pre_ping=True,
    pool_recycle=1800,
)

async_session_maker = async_sessionmaker(engine)


@asynccontextmanager
async def async_session() -> AsyncGenerator[AsyncSession]:
    """Контекстный менеджер для работы с асинхронной сессией."""
    async with async_session_maker() as session:
        yield session



class Base(AsyncAttrs, DeclarativeBase):
    """Базовый класс для всех моделей, поддерживающий асинхронные атрибуты."""


class GuildAIConfig(Base):
    """Полная provider-agnostic конфигурация AI для одной Discord-гильдии."""

    __tablename__ = "guild_ai_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    guild_id = Column(BigInteger, nullable=False, unique=True, index=True)
    provider = Column(String(50), nullable=False)
    api_url = Column(String(500), nullable=False)
    token = Column(String(1000), nullable=False)
    chat_model = Column(String(255), nullable=False)
    mini_model = Column(String(255), nullable=False)
    embedding_provider = Column(String(50), nullable=False)
    embedding_api_url = Column(String(500), nullable=False)
    embedding_token = Column(String(1000), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (Index("idx_guild_ai_config_guild", "guild_id"),)

    def __repr__(self) -> str:
        """Не включает chat/embedding credentials в диагностику ORM."""
        return f"<GuildAIConfig(guild_id={self.guild_id})>"


class ChannelMessage(Base):
    """Модель сообщения в канале Discord."""

    __tablename__ = "channel_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(BigInteger, nullable=False)
    message_id = Column(BigInteger, nullable=False)
    author = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now())


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


class GuildEmoji(Base):
    """Модель описания кастомного эмодзи для конкретного сервера."""

    __tablename__ = "guild_emojis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(32), nullable=False)
    description = Column(Text, nullable=False)
    guild_id = Column(BigInteger, nullable=False)
    datetime_insert = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint("name", "guild_id", name="uq_guild_emoji_per_guild"),
        Index("idx_guild_emoji_guild", "guild_id"),
    )


class UserDescription(Base):
    """Модель описания пользователя Discord.

    Хранит ник пользователя, его описание и привязку к серверу.
    """

    __tablename__ = "user_descriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nick = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    guild_id = Column(BigInteger, nullable=False)
    datetime_insert = Column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint("nick", "guild_id", name="uq_user_description_per_guild"),
        Index("idx_user_desc_guild", "guild_id"),
    )

    def __repr__(self) -> str:
        """Строковое представление модели UserDescription."""
        return f"<UserDescription(nick='{self.nick}', guild_id={self.guild_id})>"


class AIModel(Base):
    """Справочник допустимых имён AI-моделей."""

    __tablename__ = "ai_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)


class AIProvider(Base):
    """Справочник AI-провайдеров (имя, адрес, токен)."""

    __tablename__ = "ai_providers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    api_url = Column(String(500), nullable=False)
    token = Column(String(1000), nullable=False)


class BotAdmin(Base):
    """Администратор бота, назначенный для конкретного Discord-сервера."""

    __tablename__ = "bot_admins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    guild_id = Column(BigInteger, nullable=False)
    user_id = Column(BigInteger, nullable=False)
    username = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("guild_id", "user_id", name="uq_bot_admin_per_guild"),
        Index("idx_bot_admin_guild", "guild_id"),
    )


async def init_models() -> None:
    """Создает таблицы в базе данных, если они не существуют."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
