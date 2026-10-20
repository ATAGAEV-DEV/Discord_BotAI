"""Per-guild cache сохранённых AI-настроек."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AISettings
from app.data import models
from app.data.repositories import guild_ai_config

_cache: dict[int, AISettings] = {}
_locks: dict[int, asyncio.Lock] = {}
_loaded = False


def _lock_for(guild_id: int) -> asyncio.Lock:
    """Возвращает lock кэша для одной гильдии."""
    return _locks.setdefault(guild_id, asyncio.Lock())


def get(guild_id: int) -> AISettings | None:
    """Возвращает внутренние настройки гильдии или ``None``."""
    return _cache.get(guild_id)


def get_required(guild_id: int) -> AISettings:
    """Возвращает настройки или безопасную ошибку fail-closed."""
    settings = get(guild_id)
    if settings is None:
        raise guild_ai_config.GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не загружена")
    return settings


def get_all() -> dict[int, AISettings]:
    """Возвращает копию отображения настроек."""
    return dict(_cache)


def is_loaded() -> bool:
    """Показывает, был ли выполнен полный load_all."""
    return _loaded


async def load_all(session: AsyncSession) -> None:
    """Загружает только валидные строки из БД и не создаёт новые записи."""
    global _loaded
    loaded = await guild_ai_config.load_all(session)
    _cache.clear()
    _cache.update(loaded)
    _loaded = True


async def ensure(
    session: AsyncSession,
    guild_id: int,
    environment_settings: AISettings | Mapping[str, str | None] | None = None,
) -> AISettings:
    """Возвращает cache hit либо выполняет DB ensure под per-guild lock."""
    cached = _cache.get(guild_id)
    if cached is not None:
        return cached

    async with _lock_for(guild_id):
        cached = _cache.get(guild_id)
        if cached is not None:
            return cached
        settings = await guild_ai_config.ensure(session, guild_id, environment_settings)
        _cache[guild_id] = settings
        return settings


def invalidate(guild_id: int) -> None:
    """Удаляет только запись одной гильдии из config cache."""
    _cache.pop(guild_id, None)


def clear() -> None:
    """Очищает настройки и свободные locks."""
    global _loaded
    _cache.clear()
    _loaded = False
    for guild_id, lock in list(_locks.items()):
        if not lock.locked():
            _locks.pop(guild_id, None)
    guild_ai_config.clear_locks()
