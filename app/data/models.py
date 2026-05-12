import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
DATABASE_URL_LOCAL: str = os.getenv("DATABASE_URL_LOCAL", "")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL не задан в переменных окружения. "
        "Установите переменную для подключения к удаленному PostgreSQL."
    )

if not DATABASE_URL_LOCAL:
    raise RuntimeError(
        "DATABASE_URL_LOCAL не задан в переменных окружения. "
        "Установите переменную для подключения к локальному PostgreSQL."
    )
SCHEMA = "discord"


def get_engine(schema: str, db_url: str) -> AsyncEngine:
    """Создаёт и возвращает асинхронный движок SQLAlchemy с указанным схемой.

    Устанавливает параметр search_path в соединении, чтобы все запросы выполнялись
    в заданной схеме PostgreSQL.
    """
    return create_async_engine(
        db_url,
        connect_args={"server_settings": {"search_path": schema}},
        pool_pre_ping=True,
        pool_recycle=1800,
    )


if SCHEMA is None or SCHEMA == "":
    engine_remote = get_engine("public", DATABASE_URL)
    engine_local = get_engine("public", DATABASE_URL_LOCAL)
else:
    engine_remote = get_engine(SCHEMA, DATABASE_URL)
    engine_local = get_engine(SCHEMA, DATABASE_URL_LOCAL)

async_session_local_maker = async_sessionmaker(engine_local)
async_session_remote_maker = async_sessionmaker(engine_remote)


class DualSessionProxy:
    """Прокси-класс для выполнения операций в двух базах данных.

    Чтение выполняется только из локальной БД.
    Запись (INSERT, UPDATE, DELETE) дублируется в обе БД.
    """

    def __init__(self, local_session: AsyncSession, remote_session: AsyncSession) -> None:
        """Инициализирует прокси-сессию."""
        self.local = local_session
        self.remote = remote_session

    async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполняет запрос. DML-запросы дублируются в удаленную БД."""
        if getattr(statement, "is_dml", False):
            try:
                await self.remote.execute(statement, *args, **kwargs)
            except Exception as e:
                print(f"Ошибка выполнения запроса в удаленной БД: {e}")
        return await self.local.execute(statement, *args, **kwargs)

    def add(self, instance: Any) -> None:
        """Добавляет объект в обе сессии."""
        self.local.add(instance)
        state = instance.__dict__.copy()
        state.pop("_sa_instance_state", None)
        remote_instance = instance.__class__(**state)
        self.remote.add(remote_instance)

    async def commit(self) -> None:
        """Фиксирует транзакцию в обеих БД."""
        await self.local.commit()
        try:
            await self.remote.commit()
        except Exception as e:
            print(f"Ошибка коммита в удаленной БД: {e}")
            await self.remote.rollback()

    async def rollback(self) -> None:
        """Откатывает транзакцию в обеих БД."""
        await self.local.rollback()
        await self.remote.rollback()


@asynccontextmanager
async def async_session() -> AsyncGenerator[DualSessionProxy]:
    """Контекстный менеджер для работы с прокси-сессией DualSessionProxy."""
    async with (
        async_session_local_maker() as local_session,
        async_session_remote_maker() as remote_session,
    ):
        yield DualSessionProxy(local_session, remote_session)


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


class UserDescription(Base):
    """Модель описания пользователя Discord.

    Хранит ник пользователя, его описание и привязку к серверу.
    Заменяет захардкоженный словарь USER_DESCRIPTIONS.
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


async def init_models() -> None:
    """Создает таблицы в базах данных, если они не существуют."""
    async with engine_local.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with engine_remote.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"Ошибка создания таблиц в удаленной БД: {e}")
