"""Изоляция каждого теста отдельной схемой в каждой из двух тестовых БД."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from conftest import TEST_URLS
from tests.postgres_support import ACTIVE_SCHEMAS

if TYPE_CHECKING:
    from app.data import models


@dataclass
class Databases:
    """Независимые сессии для проверки фактически зафиксированных данных."""

    local: async_sessionmaker[AsyncSession]
    remote: async_sessionmaker[AsyncSession]
    engines: tuple[AsyncEngine, AsyncEngine]
    schema: str

    async def rows(self, side: str, model: type[models.Base]) -> list[Any]:
        """Читает таблицу новой сессией, не используя DualSessionProxy."""
        async with getattr(self, side)() as session:
            return list((await session.scalars(select(model))).all())

    async def seed(self, side: str, *rows: models.Base) -> None:
        """Создаёт намеренно независимые данные в одной БД."""
        async with getattr(self, side)() as session:
            session.add_all(rows)
            await session.commit()


def _engine(url: URL, schema: str) -> AsyncEngine:
    """Ограничивает соединения и время выполнения на общих серверах."""
    return create_async_engine(
        url,
        pool_size=2,  # YouTube держит внешнюю сессию во время вложенной операции.
        max_overflow=0,
        pool_timeout=5,
        connect_args={
            "timeout": 5,
            "command_timeout": 20,
            "server_settings": {
                "search_path": schema,
                "application_name": "discord_botai_pytest",
                "statement_timeout": "5000",
                "lock_timeout": "1000",
                "idle_in_transaction_session_timeout": "10000",
            },
        },
    )


def _reset_caches() -> None:
    """Не позволяет кэшам переходить между тестами."""
    from app.data import admins, emoji_descriptions_cache, user_descriptions_cache

    for module in (admins, emoji_descriptions_cache, user_descriptions_cache):
        module._cache.clear()
    user_descriptions_cache._loaded = False


@pytest.fixture(autouse=True)
async def databases(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Databases]:
    """Создаёт таблицы, перенаправляет приложение и удаляет только свои схемы."""
    from app.data import models

    if not request.config.getoption("--run-integration"):
        pytest.fail("Для реальных БД требуется --run-integration")
    urls = request.config.stash[TEST_URLS]
    schema = f"pytest_{uuid4().hex}"
    assert re.fullmatch(r"pytest_[0-9a-f]{32}", schema)
    request.node.user_properties.append(("postgres_schema", schema))
    engines = (
        _engine(urls["TEST_DATABASE_URL_LOCAL"], schema),
        _engine(urls["TEST_DATABASE_URL"], schema),
    )
    created: list[AsyncEngine] = []
    sessions: list[AsyncSession] = []
    ACTIVE_SCHEMAS.add(schema)
    _reset_caches()

    class TrackedSession(AsyncSession):
        """Гарантирует закрытие даже сессии, забытой упавшим тестом."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            sessions.append(self)

    try:
        for engine in engines:
            async with engine.begin() as conn:
                assert await conn.scalar(text("SELECT current_database()")) == "test_atagaev"
                await conn.execute(CreateSchema(schema))
                # Запоминаем владение до COMMIT: потеря его ответа не должна пропустить cleanup.
                created.append(engine)
            async with engine.begin() as conn:
                assert await conn.scalar(text("SELECT current_schema()")) == schema
                # DDL на общих серверах иногда занимает больше обычного SELECT/DML.
                # SET LOCAL действует только до конца setup-транзакции.
                await conn.execute(text("SET LOCAL statement_timeout = '15s'"))
                await conn.run_sync(models.Base.metadata.create_all, checkfirst=False)

        makers = tuple(async_sessionmaker(e, class_=TrackedSession) for e in engines)
        monkeypatch.setattr(models, "engine_local", engines[0])
        monkeypatch.setattr(models, "engine_remote", engines[1])
        monkeypatch.setattr(models, "async_session_local_maker", makers[0])
        monkeypatch.setattr(models, "async_session_remote_maker", makers[1])
        monkeypatch.setattr(models, "SCHEMA", schema)
        monkeypatch.setenv("DATABASE_SCHEMA", schema)
        yield Databases(makers[0], makers[1], engines, schema)
    finally:
        errors: list[Exception] = []
        for session in sessions:
            try:
                await session.close()
            except Exception as exc:
                errors.append(exc)
        for engine in engines:
            try:
                if engine in created:
                    async with engine.begin() as conn:
                        await conn.execute(DropSchema(schema, cascade=True, if_exists=True))
                        assert await conn.scalar(
                            text("SELECT to_regnamespace(:schema)"), {"schema": schema}
                        ) is None
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    await engine.dispose()
                except Exception as exc:
                    errors.append(exc)
        ACTIVE_SCHEMAS.discard(schema)
        _reset_caches()
        if errors:
            raise ExceptionGroup(f"Ошибка очистки собственной схемы {schema}", errors)