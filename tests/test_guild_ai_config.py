"""Unit-тесты repository guild_ai_configs и fail-closed bootstrap."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import AISettings
from app.data import guild_ai_config_cache
from app.data.models import GuildAIConfig
from app.data.repositories import guild_ai_config


def _settings(**overrides: object) -> AISettings:
    """Возвращает полную тестовую конфигурацию."""
    values: dict[str, object] = {
        "token": "chat-secret",
        "embedding_token": "embedding-secret",
        "api_url": "https://chat.example/v1",
        "embedding_api_url": "https://embedding.example/v1",
        "chat_model": "chat-model",
        "mini_model": "mini-model",
        "embedding_model": "embedding-model",
    }
    values.update(overrides)
    return AISettings(**values)


def _row(guild_id: int = 1, **overrides: object) -> GuildAIConfig:
    """Создаёт ORM row с обеими credentials."""
    values = {
        "guild_id": guild_id,
        "provider": "polza",
        "api_url": "https://chat.example/v1",
        "token": "stored-chat-secret",
        "chat_model": "stored-chat-model",
        "mini_model": "stored-mini-model",
        "embedding_provider": "polza",
        "embedding_api_url": "https://stored-embedding.example/v1",
        "embedding_token": "stored-embedding-secret",
        "embedding_model": "stored-embedding-model",
    }
    values.update(overrides)
    return GuildAIConfig(**values)


class _Result:
    """Минимальный SQLAlchemy result для repository tests."""

    def __init__(self, row: GuildAIConfig | None) -> None:
        self.row = row

    def scalar_one_or_none(self) -> GuildAIConfig | None:
        """Возвращает одну fake строку."""
        return self.row

    def scalars(self) -> SimpleNamespace:
        """Возвращает fake scalar result."""
        return SimpleNamespace(all=lambda: [self.row] if self.row is not None else [])


class _Side:
    """Сторона proxy-сессии с одной видимой строкой."""

    def __init__(self, owner: "_Proxy") -> None:
        self.owner = owner

    async def execute(self, statement: object) -> _Result:
        """Возвращает состояние текущей стороны."""
        del statement
        return _Result(self.owner.local_row if self.owner.local is self else self.owner.remote_row)


class _Proxy:
    """Fake DualSessionProxy с отдельными local/remote строками."""

    def __init__(self, row: GuildAIConfig | None = None) -> None:
        self.local_row = row
        self.remote_row = row
        self.local = _Side(self)
        self.remote = _Side(self)
        self.pending: GuildAIConfig | None = None
        self.commit = AsyncMock(side_effect=self._commit)
        self.rollback = AsyncMock()

    async def execute(self, statement: object) -> _Result:
        """Имитирует proxy read из local database."""
        del statement
        return _Result(self.local_row)

    def add(self, instance: GuildAIConfig) -> None:
        """Запоминает pending insert без публикации до commit."""
        self.pending = instance

    async def _commit(self, *, atomic: bool = False) -> None:
        """Публикует строку только после успешного commit."""
        assert atomic is True
        if self.pending is not None:
            self.local_row = self.pending
            self.remote_row = _row(
                self.pending.guild_id,
                provider=self.pending.provider,
                api_url=self.pending.api_url,
                token=self.pending.token,
                chat_model=self.pending.chat_model,
                mini_model=self.pending.mini_model,
                embedding_provider=self.pending.embedding_provider,
                embedding_api_url=self.pending.embedding_api_url,
                embedding_token=self.pending.embedding_token,
                embedding_model=self.pending.embedding_model,
            )


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Изолирует config cache и repository locks."""
    guild_ai_config_cache.clear()
    guild_ai_config.clear_locks()
    yield
    guild_ai_config_cache.clear()
    guild_ai_config.clear_locks()


@pytest.mark.asyncio
async def test_existing_complete_row_has_priority_over_environment() -> None:
    """Полная строка БД не заменяется environment settings."""
    proxy = _Proxy(_row())

    result = await guild_ai_config.ensure(proxy, 1, _settings(token="env-chat"))

    assert result.token == "stored-chat-secret"
    assert result.embedding_token == "stored-embedding-secret"
    assert result.api_url == "https://chat.example/v1"
    proxy.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_requires_both_tokens_and_never_falls_back() -> None:
    """Нельзя создать partial row и нельзя копировать chat token."""
    proxy = _Proxy()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="embedding token"):
        await guild_ai_config.ensure(proxy, 2, _settings(embedding_token=None))

    assert proxy.pending is None
    proxy.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_is_written_to_both_databases_after_commit() -> None:
    """Bootstrap сохраняет независимые credentials в обеих копиях."""
    proxy = _Proxy()
    settings = _settings()

    result = await guild_ai_config.ensure(proxy, 3, settings)

    assert result.token == "chat-secret"
    assert result.embedding_token == "embedding-secret"
    assert proxy.local_row is not None
    assert proxy.remote_row is not None
    assert proxy.local_row.embedding_token == "embedding-secret"
    assert proxy.remote_row.embedding_token == "embedding-secret"
    proxy.commit.assert_awaited_once_with(atomic=True)


@pytest.mark.asyncio
async def test_commit_failure_does_not_publish_cache_or_partial_settings() -> None:
    """Ошибка commit не оставляет новую конфигурацию активной."""
    proxy = _Proxy()
    proxy.commit = AsyncMock(side_effect=RuntimeError("commit failed"))

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Не удалось сохранить"):
        await guild_ai_config_cache.ensure(proxy, 4, _settings())

    assert guild_ai_config_cache.get(4) is None
    assert proxy.local_row is None
    assert proxy.remote_row is None
    proxy.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_bootstrap_uses_one_guild_lock() -> None:
    """Конкурентные вызовы одной гильдии не создают две строки."""
    proxy = _Proxy()
    original_commit = proxy.commit
    calls = 0

    async def delayed_commit(*, atomic: bool = False) -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        await original_commit(atomic=atomic)

    proxy.commit = AsyncMock(side_effect=delayed_commit)
    results = await asyncio.gather(
        guild_ai_config_cache.ensure(proxy, 5, _settings()),
        guild_ai_config_cache.ensure(proxy, 5, _settings(token="other-chat")),
    )

    assert calls == 1
    assert results[0].token == results[1].token
