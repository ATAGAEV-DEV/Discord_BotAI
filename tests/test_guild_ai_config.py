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
        "provider": "custom-chat-provider",
        "token": "chat-secret",
        "embedding_token": "embedding-secret",
        "api_url": "https://chat.example/v1",
        "embedding_provider": "custom-embedding-provider",
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
        "provider": "custom-chat-provider",
        "api_url": "https://chat.example/v1",
        "token": "stored-chat-secret",
        "chat_model": "stored-chat-model",
        "mini_model": "stored-mini-model",
        "embedding_provider": "custom-embedding-provider",
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


class _Session:
    """Fake AsyncSession с одной видимой строкой."""

    def __init__(self, row: GuildAIConfig | None = None) -> None:
        self._row = row
        self.pending: GuildAIConfig | None = None
        self.commit = AsyncMock(side_effect=self._do_commit)
        self.rollback = AsyncMock()

    async def execute(self, statement: object) -> _Result:
        """Возвращает текущую строку."""
        del statement
        return _Result(self._row)

    def add(self, instance: GuildAIConfig) -> None:
        """Запоминает pending insert без публикации до commit."""
        self.pending = instance

    async def _do_commit(self) -> None:
        """Публикует строку только после успешного commit."""
        if self.pending is not None:
            self._row = self.pending
            self.pending = None



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
    session = _Session(_row())

    result = await guild_ai_config.ensure(session, 1, _settings(token="env-chat"))

    assert result.token == "stored-chat-secret"
    assert result.embedding_token == "stored-embedding-secret"
    assert result.api_url == "https://chat.example/v1"
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_requires_embedding_token() -> None:
    """Нельзя создать строку без embedding token."""
    session = _Session()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="embedding token"):
        await guild_ai_config.ensure(session, 2, _settings(embedding_token=None))

    assert session.pending is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_requires_all_models() -> None:
    """Неполная модельная конфигурация не сохраняется в БД."""
    session = _Session()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="chat_model"):
        await guild_ai_config.ensure(session, 6, _settings(chat_model=None))

    assert session.pending is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_is_written_after_commit() -> None:
    """Bootstrap сохраняет credentials после commit."""
    session = _Session()
    settings = _settings()

    result = await guild_ai_config.ensure(session, 3, settings)

    assert result.token == "chat-secret"
    assert result.embedding_token == "embedding-secret"
    assert session._row is not None
    assert session._row.embedding_token == "embedding-secret"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_commit_failure_does_not_publish_cache_or_partial_settings() -> None:
    """Ошибка commit не оставляет новую конфигурацию активной."""
    session = _Session()
    session.commit = AsyncMock(side_effect=RuntimeError("commit failed"))

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Не удалось сохранить"):
        await guild_ai_config_cache.ensure(session, 4, _settings())

    assert guild_ai_config_cache.get(4) is None
    assert session._row is None
    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_bootstrap_uses_one_guild_lock() -> None:
    """Конкурентные вызовы одной гильдии не создают две строки."""
    session = _Session()
    original_commit = session.commit
    calls = 0

    async def delayed_commit() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        await original_commit()

    session.commit = AsyncMock(side_effect=delayed_commit)
    results = await asyncio.gather(
        guild_ai_config_cache.ensure(session, 5, _settings()),
        guild_ai_config_cache.ensure(session, 5, _settings(token="other-chat")),
    )

    assert calls == 1
    assert results[0].token == results[1].token

