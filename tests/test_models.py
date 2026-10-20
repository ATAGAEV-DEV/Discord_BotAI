"""Изолированные тесты для моделей и сессии."""

import runpy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import dotenv
import pytest
import sqlalchemy.ext.asyncio as sqlalchemy_asyncio

from app.core import config
from app.data import models

MODELS_PATH = Path(__file__).parents[1] / "app" / "data" / "models.py"


def _run_models_module() -> dict[str, object]:
    """Исполняет models.py в отдельном namespace для проверки импорта."""
    return runpy.run_path(str(MODELS_PATH), run_name="app.data.models_isolated")


def test_import_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Импорт моделей завершается ошибкой при отсутствии DATABASE_URL."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL не задан"):
        _run_models_module()


@pytest.mark.parametrize("schema", [None, ""])
def test_import_uses_public_schema_when_schema_is_empty(
    monkeypatch: pytest.MonkeyPatch, schema: str | None
) -> None:
    """Пустая схема направляет engine в стандартную схему public."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://remote/test")
    monkeypatch.setattr(config, "get_database_settings", lambda: SimpleNamespace(schema=schema))

    create_engine = MagicMock(return_value=MagicMock(name="engine"))
    sessionmaker = MagicMock(return_value=MagicMock(name="maker"))
    monkeypatch.setattr(sqlalchemy_asyncio, "create_async_engine", create_engine)
    monkeypatch.setattr(sqlalchemy_asyncio, "async_sessionmaker", sessionmaker)

    isolated_module = _run_models_module()

    assert isolated_module["SCHEMA"] == schema
    create_engine.assert_called_once_with(
        "postgresql+asyncpg://remote/test",
        connect_args={"server_settings": {"search_path": "public"}},
        pool_pre_ping=True,
        pool_recycle=1800,
    )


@pytest.mark.parametrize("schema", ["myschema", "test_schema"])
def test_import_uses_explicit_schema(
    monkeypatch: pytest.MonkeyPatch, schema: str
) -> None:
    """Явно заданная схема передаётся в engine."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://remote/test")
    monkeypatch.setattr(config, "get_database_settings", lambda: SimpleNamespace(schema=schema))

    create_engine = MagicMock(return_value=MagicMock(name="engine"))
    sessionmaker = MagicMock(return_value=MagicMock(name="maker"))
    monkeypatch.setattr(sqlalchemy_asyncio, "create_async_engine", create_engine)
    monkeypatch.setattr(sqlalchemy_asyncio, "async_sessionmaker", sessionmaker)

    isolated_module = _run_models_module()

    assert isolated_module["SCHEMA"] == schema
    create_engine.assert_called_once_with(
        "postgresql+asyncpg://remote/test",
        connect_args={"server_settings": {"search_path": schema}},
        pool_pre_ping=True,
        pool_recycle=1800,
    )


@pytest.mark.asyncio
async def test_async_session_yields_session_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Контекстный менеджер возвращает AsyncSession и закрывает её."""
    session = MagicMock(name="session")
    lifecycle: list[str] = []

    @asynccontextmanager
    async def session_context() -> AsyncIterator[MagicMock]:
        lifecycle.append("enter")
        try:
            yield session
        finally:
            lifecycle.append("exit")

    monkeypatch.setattr(models, "async_session_maker", session_context)

    async with models.async_session() as s:
        assert s is session
        assert lifecycle == ["enter"]

    assert lifecycle == ["enter", "exit"]


def test_user_description_repr_contains_nick_and_guild() -> None:
    """Строковое представление содержит ник и идентификатор сервера."""
    description = models.UserDescription(
        nick="Alice",
        description="Описание",
        guild_id=100,
    )

    assert repr(description) == "<UserDescription(nick='Alice', guild_id=100)>"


def _engine_with_connection(connection: MagicMock) -> MagicMock:
    """Создаёт mock-engine с асинхронным контекстом подключения."""
    engine = MagicMock()

    @asynccontextmanager
    async def begin() -> AsyncIterator[MagicMock]:
        yield connection

    engine.begin.return_value = begin()
    return engine


@pytest.mark.asyncio
async def test_init_models_creates_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Инициализация создаёт таблицы в engine."""
    connection = MagicMock()
    connection.run_sync = AsyncMock()
    mock_engine = _engine_with_connection(connection)
    monkeypatch.setattr(models, "engine", mock_engine)

    await models.init_models()

    connection.run_sync.assert_awaited_once_with(models.Base.metadata.create_all)
    mock_engine.begin.assert_called_once_with()


def test_guild_ai_config_repr_contains_guild_id() -> None:
    """GuildAIConfig.__repr__ содержит guild_id и не раскрывает credentials."""
    cfg = models.GuildAIConfig(
        guild_id=12345,
        provider="openai",
        api_url="https://api.openai.com/v1",
        token="secret-token",
        chat_model="gpt-4",
        mini_model="gpt-4-mini",
        embedding_provider="openai",
        embedding_api_url="https://api.openai.com/v1",
        embedding_token="emb-secret",
        embedding_model="text-embedding-3-small",
    )

    result = repr(cfg)

    assert "12345" in result
    assert "GuildAIConfig" in result
    assert "secret-token" not in result
    assert "emb-secret" not in result

