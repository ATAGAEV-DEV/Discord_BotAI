"""Изолированные тесты для моделей и двухбазовой сессии."""

import runpy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import dotenv
import pytest
import sqlalchemy.ext.asyncio as sqlalchemy_asyncio

from app.core import config
from app.data import models

MODELS_PATH = Path(__file__).parents[1] / "app" / "data" / "models.py"


@pytest.fixture
def database_sessions() -> tuple[MagicMock, MagicMock]:
    """Создаёт две полностью изолированные mock-сессии."""
    local = MagicMock()
    local.execute = AsyncMock()
    local.commit = AsyncMock()
    local.rollback = AsyncMock()
    local.delete = AsyncMock()

    remote = MagicMock()
    remote.execute = AsyncMock()
    remote.commit = AsyncMock()
    remote.rollback = AsyncMock()
    remote.delete = AsyncMock()

    return local, remote


@pytest.fixture
def session_proxy(database_sessions: tuple[MagicMock, MagicMock]) -> models.DualSessionProxy:
    """Создаёт прокси над локальной и удалённой mock-сессиями."""
    local, remote = database_sessions
    return models.DualSessionProxy(local, remote)


def _run_models_module() -> dict[str, object]:
    """Исполняет models.py в отдельном namespace для проверки импорта."""
    return runpy.run_path(str(MODELS_PATH), run_name="app.data.models_isolated")


@pytest.mark.parametrize("missing_url", ["DATABASE_URL", "DATABASE_URL_LOCAL"])
def test_import_requires_both_database_urls(
    monkeypatch: pytest.MonkeyPatch, missing_url: str
) -> None:
    """Импорт моделей завершается ошибкой при отсутствии любого URL базы данных."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://remote/test")
    monkeypatch.setenv("DATABASE_URL_LOCAL", "postgresql+asyncpg://local/test")
    monkeypatch.delenv(missing_url)

    expected_variable = missing_url
    with pytest.raises(RuntimeError, match=f"{expected_variable} не задан"):
        _run_models_module()


@pytest.mark.parametrize("schema", [None, ""])
def test_import_uses_public_schema_when_schema_is_empty(
    monkeypatch: pytest.MonkeyPatch, schema: str | None
) -> None:
    """Пустая схема направляет оба engine в стандартную схему public."""
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://remote/test")
    monkeypatch.setenv("DATABASE_URL_LOCAL", "postgresql+asyncpg://local/test")
    monkeypatch.setattr(config, "get_database_settings", lambda: SimpleNamespace(schema=schema))

    create_engine = MagicMock(
        side_effect=[MagicMock(name="remote_engine"), MagicMock(name="local_engine")]
    )
    sessionmaker = MagicMock(
        side_effect=[MagicMock(name="local_maker"), MagicMock(name="remote_maker")]
    )
    monkeypatch.setattr(sqlalchemy_asyncio, "create_async_engine", create_engine)
    monkeypatch.setattr(sqlalchemy_asyncio, "async_sessionmaker", sessionmaker)

    isolated_module = _run_models_module()

    assert isolated_module["SCHEMA"] == schema
    assert create_engine.call_args_list == [
        call(
            "postgresql+asyncpg://remote/test",
            connect_args={"server_settings": {"search_path": "public"}},
            pool_pre_ping=True,
            pool_recycle=1800,
        ),
        call(
            "postgresql+asyncpg://local/test",
            connect_args={"server_settings": {"search_path": "public"}},
            pool_pre_ping=True,
            pool_recycle=1800,
        ),
    ]


def test_proxy_stores_both_sessions(
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """Прокси сохраняет ссылки на локальную и удалённую сессии."""
    local, remote = database_sessions

    proxy = models.DualSessionProxy(local, remote)

    assert proxy.local is local
    assert proxy.remote is remote


@pytest.mark.asyncio
async def test_execute_reads_only_from_local_session(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """Обычный запрос выполняется только в локальной базе."""
    local, remote = database_sessions
    statement = object()
    expected_result = object()
    local.execute.return_value = expected_result

    result = await session_proxy.execute(statement, "parameter", option=True)

    assert result is expected_result
    local.execute.assert_awaited_once_with(statement, "parameter", option=True)
    remote.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_duplicates_dml_in_remote_session(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """DML-запрос сначала выполняется в удалённой, а затем в локальной базе."""
    local, remote = database_sessions
    statement = SimpleNamespace(is_dml=True)
    expected_result = object()
    local.execute.return_value = expected_result

    result = await session_proxy.execute(statement)

    assert result is expected_result
    remote.execute.assert_awaited_once_with(statement)
    local.execute.assert_awaited_once_with(statement)


@pytest.mark.asyncio
async def test_execute_continues_locally_after_remote_error(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка DML в удалённой базе не мешает выполнению локального запроса."""
    local, remote = database_sessions
    statement = SimpleNamespace(is_dml=True)
    remote.execute.side_effect = RuntimeError("remote execute failed")
    expected_result = object()
    local.execute.return_value = expected_result

    result = await session_proxy.execute(statement)

    assert result is expected_result
    local.execute.assert_awaited_once_with(statement)
    output = capsys.readouterr().out
    assert "Ошибка выполнения запроса в удаленной БД: remote execute failed" in output


def test_add_copies_instance_to_both_sessions(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """Добавление создаёт независимый ORM-объект для удалённой сессии."""
    local, remote = database_sessions
    instance = models.UserDescription(
        id=7,
        nick="Alice",
        description="Описание",
        guild_id=100,
    )

    session_proxy.add(instance)

    local.add.assert_called_once_with(instance)
    remote.add.assert_called_once()
    remote_instance = remote.add.call_args.args[0]
    assert isinstance(remote_instance, models.UserDescription)
    assert remote_instance is not instance
    assert remote_instance.id == instance.id
    assert remote_instance.nick == instance.nick
    assert remote_instance.description == instance.description
    assert remote_instance.guild_id == instance.guild_id


@pytest.mark.asyncio
async def test_commit_commits_both_sessions(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """Успешный commit выполняется в локальной и удалённой базах."""
    local, remote = database_sessions

    await session_proxy.commit()

    local.commit.assert_awaited_once()
    remote.commit.assert_awaited_once()
    remote.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_rolls_back_remote_session_after_error(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка удалённого commit вызывает rollback удалённой сессии."""
    local, remote = database_sessions
    remote.commit.side_effect = RuntimeError("remote commit failed")

    await session_proxy.commit()

    local.commit.assert_awaited_once()
    remote.commit.assert_awaited_once()
    remote.rollback.assert_awaited_once()
    assert "Ошибка коммита в удаленной БД: remote commit failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_rollback_rolls_back_both_sessions(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """Rollback выполняется в обеих базах."""
    local, remote = database_sessions

    await session_proxy.rollback()

    local.rollback.assert_awaited_once()
    remote.rollback.assert_awaited_once()


def _scalar_result(value: object) -> MagicMock:
    """Создаёт mock-результат SQLAlchemy с одним объектом или None."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_delete_removes_instance_from_both_sessions(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """Удаление ищет и удаляет соответствующую запись в удалённой базе."""
    local, remote = database_sessions
    instance = models.UserDescription(
        id=7,
        nick="Alice",
        description="Описание",
        guild_id=100,
    )
    remote_instance = models.UserDescription(
        id=7,
        nick="Alice",
        description="Описание",
        guild_id=100,
    )
    remote.execute.return_value = _scalar_result(remote_instance)

    await session_proxy.delete(instance)

    local.delete.assert_awaited_once_with(instance)
    remote.execute.assert_awaited_once()
    remote.delete.assert_awaited_once_with(remote_instance)


@pytest.mark.asyncio
async def test_delete_does_not_delete_missing_remote_instance(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
) -> None:
    """При отсутствии записи в удалённой базе дополнительное удаление не выполняется."""
    local, remote = database_sessions
    instance = models.UserDescription(
        id=7,
        nick="Alice",
        description="Описание",
        guild_id=100,
    )
    remote.execute.return_value = _scalar_result(None)

    await session_proxy.delete(instance)

    local.delete.assert_awaited_once_with(instance)
    remote.execute.assert_awaited_once()
    remote.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_logs_remote_error_and_keeps_local_delete(
    session_proxy: models.DualSessionProxy,
    database_sessions: tuple[MagicMock, MagicMock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка удалённой базы логируется после успешного локального удаления."""
    local, remote = database_sessions
    instance = models.UserDescription(
        id=7,
        nick="Alice",
        description="Описание",
        guild_id=100,
    )
    remote.execute.side_effect = RuntimeError("remote delete failed")

    await session_proxy.delete(instance)

    local.delete.assert_awaited_once_with(instance)
    assert "Ошибка удаления из удалённой БД: remote delete failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_async_session_yields_proxy_and_closes_both_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Контекстный менеджер объединяет и закрывает обе сессии."""
    local = MagicMock(name="local_session")
    remote = MagicMock(name="remote_session")
    lifecycle: list[str] = []

    @asynccontextmanager
    async def local_context() -> AsyncIterator[MagicMock]:
        lifecycle.append("local-enter")
        try:
            yield local
        finally:
            lifecycle.append("local-exit")

    @asynccontextmanager
    async def remote_context() -> AsyncIterator[MagicMock]:
        lifecycle.append("remote-enter")
        try:
            yield remote
        finally:
            lifecycle.append("remote-exit")

    monkeypatch.setattr(models, "async_session_local_maker", local_context)
    monkeypatch.setattr(models, "async_session_remote_maker", remote_context)

    async with models.async_session() as session:
        assert isinstance(session, models.DualSessionProxy)
        assert session.local is local
        assert session.remote is remote
        assert lifecycle == ["local-enter", "remote-enter"]

    assert lifecycle == ["local-enter", "remote-enter", "remote-exit", "local-exit"]


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
async def test_init_models_creates_tables_in_both_databases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Инициализация создаёт таблицы в локальном и удалённом engine."""
    local_connection = MagicMock()
    local_connection.run_sync = AsyncMock()
    remote_connection = MagicMock()
    remote_connection.run_sync = AsyncMock()
    local_engine = _engine_with_connection(local_connection)
    remote_engine = _engine_with_connection(remote_connection)
    monkeypatch.setattr(models, "engine_local", local_engine)
    monkeypatch.setattr(models, "engine_remote", remote_engine)

    await models.init_models()

    local_connection.run_sync.assert_awaited_once_with(models.Base.metadata.create_all)
    remote_connection.run_sync.assert_awaited_once_with(models.Base.metadata.create_all)
    local_engine.begin.assert_called_once_with()
    remote_engine.begin.assert_called_once_with()


@pytest.mark.asyncio
async def test_init_models_logs_remote_creation_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка создания таблиц в удалённом engine не пробрасывается наружу."""
    local_connection = MagicMock()
    local_connection.run_sync = AsyncMock()
    remote_connection = MagicMock()
    remote_connection.run_sync = AsyncMock(side_effect=RuntimeError("remote init failed"))
    monkeypatch.setattr(models, "engine_local", _engine_with_connection(local_connection))
    monkeypatch.setattr(models, "engine_remote", _engine_with_connection(remote_connection))

    result = await models.init_models()

    assert result is None
    assert "Ошибка создания таблиц в удаленной БД: remote init failed" in capsys.readouterr().out