"""Unit-тесты per-guild cache конфигураций."""

from unittest.mock import AsyncMock

import pytest

from app.core.config import AISettings
from app.data import guild_ai_config_cache
from app.data.repositories import guild_ai_config


def _settings(guild_id: int) -> AISettings:
    """Создаёт настройки cache entry."""
    return AISettings(
        provider="custom-chat-provider",
        token=f"chat-{guild_id}",
        embedding_token=f"embedding-{guild_id}",
        api_url=f"https://chat-{guild_id}.example/v1",
        embedding_provider="custom-embedding-provider",
        embedding_api_url=f"https://embedding-{guild_id}.example/v1",
    )


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Очищает cache до и после каждого теста."""
    guild_ai_config_cache.clear()
    yield
    guild_ai_config_cache.clear()


@pytest.mark.asyncio
async def test_cache_isolated_by_guild_and_hit_skips_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Разные guild_id не смешиваются, повторный hit не вызывает DB."""
    ensure = AsyncMock(
        side_effect=lambda session, guild_id, environment_settings=None: _settings(guild_id)
    )
    monkeypatch.setattr(guild_ai_config, "ensure", ensure)

    first = await guild_ai_config_cache.ensure(object(), 10)
    second = await guild_ai_config_cache.ensure(object(), 20)
    again = await guild_ai_config_cache.ensure(object(), 10)

    assert first.token == "chat-10"
    assert second.embedding_token == "embedding-20"
    assert again is first
    assert ensure.await_count == 2


@pytest.mark.asyncio
async def test_load_all_replaces_stale_entries_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_all только загружает БД и очищает stale cache."""
    first, second = _settings(1), _settings(2)
    loader = AsyncMock(return_value={1: first, 2: second})
    monkeypatch.setattr(guild_ai_config, "load_all", loader)
    guild_ai_config_cache._cache[999] = _settings(999)

    await guild_ai_config_cache.load_all(object())

    assert guild_ai_config_cache.is_loaded()
    assert guild_ai_config_cache.get_all() == {1: first, 2: second}
    loader.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_error_does_not_change_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ошибка bootstrap не публикует неподтверждённые settings."""
    monkeypatch.setattr(
        guild_ai_config,
        "ensure",
        AsyncMock(side_effect=guild_ai_config.GuildAIConfigError("invalid")),
    )

    with pytest.raises(guild_ai_config.GuildAIConfigError):
        await guild_ai_config_cache.ensure(object(), 10)

    assert guild_ai_config_cache.get(10) is None


def test_clear_removes_entries_and_loaded_state() -> None:
    """Clear полностью сбрасывает config cache."""
    guild_ai_config_cache._cache[1] = _settings(1)
    guild_ai_config_cache._loaded = True

    guild_ai_config_cache.clear()

    assert guild_ai_config_cache.get_all() == {}
    assert not guild_ai_config_cache.is_loaded()


@pytest.mark.asyncio
async def test_ensure_double_check_cache_hit_inside_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure выполняет double-check внутри lock и возвращает уже заполненный cache hit."""
    import asyncio

    db_ensure_calls = 0
    original_settings = _settings(42)

    async def _slow_ensure(session: object, guild_id: int, env: object = None) -> AISettings:
        nonlocal db_ensure_calls
        db_ensure_calls += 1
        await asyncio.sleep(0)  # уступаем управление
        guild_ai_config_cache._cache[guild_id] = original_settings
        return original_settings

    monkeypatch.setattr(guild_ai_config, "ensure", _slow_ensure)

    # Первый вызов заходит в lock и запускает _slow_ensure.
    # Пока он ждёт, второй вызов тоже проходит первый cache miss,
    # занимает очередь на lock и при входе уже видит заполненный cache.
    results = await asyncio.gather(
        guild_ai_config_cache.ensure(object(), 42),
        guild_ai_config_cache.ensure(object(), 42),
    )

    # DB должна быть вызвана ровно один раз — второй вызов поймал double-check
    assert db_ensure_calls == 1
    assert results[0] is original_settings
    assert results[1] is original_settings
