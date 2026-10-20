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