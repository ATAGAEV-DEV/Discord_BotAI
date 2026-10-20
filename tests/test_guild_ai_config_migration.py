"""Unit-тесты provider-agnostic backfill guild AI-конфигурации."""

from types import SimpleNamespace
from typing import Any

import pytest

from app.data.migrations import guild_ai_config


class _Result:
    """Минимальный результат SQLAlchemy для migration backfill."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> "_Result":
        """Возвращает mapping view результата."""
        return self

    def all(self) -> list[dict[str, Any]]:
        """Возвращает тестовые строки."""
        return self._rows


class _Connection:
    """Fake connection для проверки SQL-параметров backfill."""

    dialect = SimpleNamespace(name="sqlite")

    def __init__(self, initial: dict[str, Any], final: dict[str, Any]) -> None:
        self._results = [_Result([initial]), _Result([final])]
        self.updates: list[dict[str, Any]] = []

    async def execute(self, statement: object, params: dict[str, Any] | None = None) -> _Result:
        """Возвращает select result или сохраняет параметры UPDATE."""
        del statement
        if params is not None:
            self.updates.append(params)
            return _Result([])
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_backfill_preserves_provider_and_falls_back_to_chat_settings() -> None:
    """Пустые embedding provider и URL наследуют произвольные chat-значения."""
    initial = {
        "guild_id": 1,
        "provider": "custom-chat-provider",
        "api_url": "https://chat.example/v1",
        "token": "chat-secret",
        "chat_model": "chat-model",
        "mini_model": "mini-model",
        "embedding_provider": None,
        "embedding_api_url": None,
        "embedding_token": None,
        "embedding_model": "embedding-model",
    }
    final = {
        **initial,
        "embedding_provider": "custom-chat-provider",
        "embedding_api_url": "https://chat.example/v1",
        "embedding_token": "embedding-secret",
    }
    connection = _Connection(initial, final)

    snapshot, changed = await guild_ai_config._backfill(
        connection,
        secret_mapping=None,
        url_mapping=None,
        migration_token="embedding-secret",
        migration_url=None,
    )

    assert changed == {1}
    assert {"embedding_provider": "custom-chat-provider", "guild_id": 1} in connection.updates
    assert {"embedding_api_url": "https://chat.example/v1", "guild_id": 1} in connection.updates
    assert snapshot[1][1] == "custom-chat-provider"
    assert snapshot[1][6] == "custom-chat-provider"
    assert snapshot[1][7] == "https://chat.example/v1"


# ---------------------------------------------------------------------------
# _backfill — missing_secret, missing_url, incomplete snapshot (130, 142, 154, 159)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_raises_when_missing_secret() -> None:
    """_backfill выбрасывает ошибку если embedding_token отсутствует и токен не передан."""
    initial = {
        "guild_id": 5, "provider": "p", "api_url": "https://p.example/v1",
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": "p", "embedding_api_url": "https://p.example/v1",
        "embedding_token": None, "embedding_model": "embed-v1",
    }
    final = {**initial, "embedding_token": None}
    connection = _Connection(initial, final)

    with pytest.raises(guild_ai_config.GuildAIConfigMigrationError, match="embedding secret"):
        await guild_ai_config._backfill(
            connection, secret_mapping=None, url_mapping=None,
            migration_token=None, migration_url=None,
        )


@pytest.mark.asyncio
async def test_backfill_uses_chat_api_url_as_embedding_url_fallback() -> None:
    """_backfill использует chat api_url как fallback для embedding_api_url."""
    initial = {
        "guild_id": 6, "provider": "p", "api_url": "https://p.example/v1",
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": "p", "embedding_api_url": None,
        "embedding_token": "emb-tok", "embedding_model": "embed-v1",
    }
    final = {**initial, "embedding_api_url": "https://p.example/v1"}
    connection = _Connection(initial, final)

    snapshot, changed = await guild_ai_config._backfill(
        connection, secret_mapping=None, url_mapping=None,
        migration_token=None, migration_url=None,
    )

    assert 6 in changed
    assert {"embedding_api_url": "https://p.example/v1", "guild_id": 6} in connection.updates


@pytest.mark.asyncio
async def test_backfill_raises_on_incomplete_final_snapshot() -> None:
    """_backfill выбрасывает ошибку если финальный snapshot содержит пустые поля."""
    initial = {
        "guild_id": 7, "provider": "p", "api_url": "https://p.example/v1",
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": "p", "embedding_api_url": "https://p.example/v1",
        "embedding_token": "emb-tok", "embedding_model": "embed-v1",
    }
    final = {**initial, "chat_model": ""}  # пустая строка
    connection = _Connection(initial, final)

    with pytest.raises(guild_ai_config.GuildAIConfigMigrationError, match="неполная конфигурация"):
        await guild_ai_config._backfill(
            connection, secret_mapping=None, url_mapping=None,
            migration_token="emb-tok", migration_url=None,
        )


@pytest.mark.asyncio
async def test_backfill_no_changes_for_complete_config() -> None:
    """_backfill не делает UPDATE если все поля уже заполнены."""
    initial = {
        "guild_id": 8, "provider": "p", "api_url": "https://p.example/v1",
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": "custom-embed", "embedding_api_url": "https://embed.example/v1",
        "embedding_token": "emb-tok", "embedding_model": "embed-v1",
    }
    connection = _Connection(initial, dict(initial))

    snapshot, changed = await guild_ai_config._backfill(
        connection, secret_mapping=None, url_mapping=None,
        migration_token=None, migration_url=None,
    )

    assert changed == set()


# ---------------------------------------------------------------------------
# _backfill — missing_url когда api_url тоже None (строки 142, 159)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backfill_raises_missing_url_when_api_url_is_none() -> None:
    """_backfill добавляет в missing_url если embedding_api_url=None и api_url=None."""
    initial = {
        "guild_id": 9, "provider": "p", "api_url": None,  # api_url тоже None
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": "p", "embedding_api_url": None,
        "embedding_token": "emb-tok", "embedding_model": "embed-v1",
    }
    final = dict(initial)
    connection = _Connection(initial, final)

    with pytest.raises(guild_ai_config.GuildAIConfigMigrationError, match="embedding URL"):
        await guild_ai_config._backfill(
            connection, secret_mapping=None, url_mapping=None,
            migration_token="emb-tok", migration_url=None,  # URL не передан
        )


# ---------------------------------------------------------------------------
# _set_not_null — PostgreSQL (строки 197-198)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_not_null_executes_alter_for_postgresql() -> None:
    """_set_not_null выполняет ALTER TABLE для PostgreSQL."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="postgresql")
    connection.execute = AsyncMock()

    await guild_ai_config._set_not_null(connection)

    assert connection.execute.await_count == 3
    executed_stmts = [str(call[0][0]) for call in connection.execute.call_args_list]
    assert any("embedding_provider" in s for s in executed_stmts)
    assert any("embedding_api_url" in s for s in executed_stmts)
    assert any("embedding_token" in s for s in executed_stmts)


@pytest.mark.asyncio
async def test_set_not_null_skips_for_non_postgresql() -> None:
    """_set_not_null не выполняет ничего для SQLite."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="sqlite")
    connection.execute = AsyncMock()

    await guild_ai_config._set_not_null(connection)

    connection.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# _table_columns (строка 43)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_table_columns_returns_column_names() -> None:
    """_table_columns возвращает имена колонок через run_sync."""
    from unittest.mock import AsyncMock, MagicMock

    columns_result = {"col1", "col2", "col3"}
    connection = MagicMock()
    connection.run_sync = AsyncMock(return_value=columns_result)

    result = await guild_ai_config._table_columns(connection)

    assert result == columns_result
    connection.run_sync.assert_awaited_once()


# ---------------------------------------------------------------------------
# _prepare_schema (строки 52-63)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prepare_schema_adds_missing_columns_for_sqlite() -> None:
    """_prepare_schema добавляет новые колонки для SQLite без IF NOT EXISTS."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    existing_columns = {
        "guild_id", "provider", "api_url", "token",
        "chat_model", "mini_model", "embedding_model",
    }
    call_count = 0

    async def _run_sync(fn: object) -> object:
        nonlocal call_count
        call_count += 1
        return existing_columns

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="sqlite")
    connection.run_sync = AsyncMock(side_effect=_run_sync)
    connection.execute = AsyncMock()

    result = await guild_ai_config._prepare_schema(connection)

    assert result == existing_columns
    assert connection.execute.await_count == 3
    for call_args in connection.execute.call_args_list:
        stmt_str = str(call_args[0][0])
        assert "IF NOT EXISTS" not in stmt_str


@pytest.mark.asyncio
async def test_prepare_schema_uses_if_not_exists_for_postgresql() -> None:
    """_prepare_schema использует IF NOT EXISTS синтаксис для PostgreSQL."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    existing_columns = {
        "guild_id", "provider", "api_url", "token",
        "chat_model", "mini_model", "embedding_model",
    }

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="postgresql")
    connection.run_sync = AsyncMock(return_value=existing_columns)
    connection.execute = AsyncMock()

    await guild_ai_config._prepare_schema(connection)

    assert connection.execute.await_count == 3
    for call_args in connection.execute.call_args_list:
        stmt_str = str(call_args[0][0])
        assert "IF NOT EXISTS" in stmt_str


@pytest.mark.asyncio
async def test_prepare_schema_skips_existing_columns() -> None:
    """_prepare_schema не добавляет колонки, которые уже существуют."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    all_columns = {
        "guild_id", "provider", "api_url", "token",
        "chat_model", "mini_model", "embedding_model",
        "embedding_provider", "embedding_api_url", "embedding_token",
    }

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="sqlite")
    connection.run_sync = AsyncMock(return_value=all_columns)
    connection.execute = AsyncMock()

    await guild_ai_config._prepare_schema(connection)

    connection.execute.assert_not_awaited()



# ---------------------------------------------------------------------------
# migrate() (строки 203-255)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_migrate_commits_and_returns_migration_result() -> None:
    """migrate() успешно выполняет backfill и возвращает MigrationResult."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    complete_row = {
        "guild_id": 1, "provider": "p", "api_url": "https://p.example/v1",
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": "custom-embed", "embedding_api_url": "https://embed.example/v1",
        "embedding_token": "emb-tok", "embedding_model": "embed-v1",
    }

    execute_calls = 0

    async def _fake_execute(stmt: object, params: object = None) -> _Result:
        nonlocal execute_calls
        execute_calls += 1
        if params is not None:
            return _Result([])
        return _Result([complete_row])

    all_columns = {
        "guild_id", "provider", "api_url", "token", "chat_model", "mini_model",
        "embedding_provider", "embedding_api_url", "embedding_token", "embedding_model",
    }

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="sqlite")
    connection.execute = AsyncMock(side_effect=_fake_execute)
    connection.run_sync = AsyncMock(return_value=all_columns)

    transaction = MagicMock()
    transaction.commit = AsyncMock()
    transaction.rollback = AsyncMock()

    connection.begin = AsyncMock(return_value=transaction)
    connection.close = AsyncMock()

    engine = MagicMock()
    engine.connect = AsyncMock(return_value=connection)

    result = await guild_ai_config.migrate(
        migration_token="emb-tok",
        migration_url=None,
        db_engine=engine,
    )

    transaction.commit.assert_awaited_once()
    transaction.rollback.assert_not_awaited()
    connection.close.assert_awaited_once()
    assert result.guild_count == 1
    assert result.changed is False
    assert result.backfilled_guilds == ()


@pytest.mark.asyncio
async def test_migrate_rollback_on_unexpected_exception() -> None:
    """migrate() делает rollback и оборачивает неожиданное исключение."""
    from unittest.mock import AsyncMock, MagicMock

    connection = MagicMock()
    connection.run_sync = AsyncMock(side_effect=RuntimeError("db error"))

    transaction = MagicMock()
    transaction.rollback = AsyncMock()
    transaction.commit = AsyncMock()

    connection.begin = AsyncMock(return_value=transaction)
    connection.close = AsyncMock()

    engine = MagicMock()
    engine.connect = AsyncMock(return_value=connection)

    with pytest.raises(guild_ai_config.GuildAIConfigMigrationError, match="не выполнена"):
        await guild_ai_config.migrate(db_engine=engine)

    transaction.rollback.assert_awaited_once()
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_migrate_reraises_migration_error_directly() -> None:
    """migrate() перебрасывает GuildAIConfigMigrationError без обёртки."""
    from unittest.mock import AsyncMock, MagicMock

    connection = MagicMock()
    connection.run_sync = AsyncMock(
        side_effect=guild_ai_config.GuildAIConfigMigrationError("прямая ошибка")
    )

    transaction = MagicMock()
    transaction.rollback = AsyncMock()
    transaction.commit = AsyncMock()

    connection.begin = AsyncMock(return_value=transaction)
    connection.close = AsyncMock()

    engine = MagicMock()
    engine.connect = AsyncMock(return_value=connection)

    with pytest.raises(guild_ai_config.GuildAIConfigMigrationError, match="прямая ошибка"):
        await guild_ai_config.migrate(db_engine=engine)

    transaction.rollback.assert_awaited_once()



@pytest.mark.asyncio
async def test_migrate_invalidates_changed_guilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """migrate() вызывает invalidate для гильдий с изменёнными настройками."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    initial_row = {
        "guild_id": 10, "provider": "p", "api_url": "https://p.example/v1",
        "token": "chat-tok", "chat_model": "gpt-4", "mini_model": "gpt-mini",
        "embedding_provider": None,
        "embedding_api_url": "https://p.example/v1",
        "embedding_token": "emb-tok",
        "embedding_model": "embed-v1",
    }
    final_row = {**initial_row, "embedding_provider": "p"}

    call_num = 0

    async def _fake_execute(stmt: object, params: object = None) -> _Result:
        nonlocal call_num
        call_num += 1
        if params is not None:
            return _Result([])
        if call_num == 1:
            return _Result([initial_row])
        return _Result([final_row])

    all_columns = {
        "guild_id", "provider", "api_url", "token", "chat_model", "mini_model",
        "embedding_provider", "embedding_api_url", "embedding_token", "embedding_model",
    }

    connection = MagicMock()
    connection.dialect = SimpleNamespace(name="sqlite")
    connection.execute = AsyncMock(side_effect=_fake_execute)
    connection.run_sync = AsyncMock(return_value=all_columns)

    transaction = MagicMock()
    transaction.commit = AsyncMock()
    transaction.rollback = AsyncMock()
    connection.begin = AsyncMock(return_value=transaction)
    connection.close = AsyncMock()

    engine = MagicMock()
    engine.connect = AsyncMock(return_value=connection)

    invalidated_guilds: list[int] = []
    invalidate_embedding_calls: list[int] = []

    from app.core import ai_config as ai_config_module
    from app.data import guild_ai_config_cache as cache_module

    monkeypatch.setattr(cache_module, "invalidate", lambda gid: invalidated_guilds.append(gid))
    monkeypatch.setattr(
        ai_config_module, "invalidate_embedding_runtime",
        lambda gid: invalidate_embedding_calls.append(gid),
    )

    result = await guild_ai_config.migrate(
        migration_token="emb-tok",
        migration_url=None,
        db_engine=engine,
    )

    assert result.changed is True
    assert 10 in result.backfilled_guilds
    assert 10 in invalidated_guilds
    assert 10 in invalidate_embedding_calls

