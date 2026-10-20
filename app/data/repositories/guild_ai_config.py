"""Repository полной AI-конфигурации конкретной Discord-гильдии."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.config import AISettings, get_ai_settings, validate_ai_settings
from app.data.models import GuildAIConfig

if TYPE_CHECKING:
    from app.data.models import DualSessionProxy


class GuildAIConfigError(ValueError):
    """Безопасная ошибка конфигурации без включения credentials."""


_locks: dict[int, asyncio.Lock] = {}


def _lock_for(guild_id: int) -> asyncio.Lock:
    """Возвращает lock, изолированный по гильдии."""
    return _locks.setdefault(guild_id, asyncio.Lock())


def _row_to_settings(row: GuildAIConfig) -> AISettings:
    """Преобразует ORM-строку во внутренние настройки."""
    settings = AISettings(
        provider=row.provider,
        api_url=row.api_url,
        token=row.token,
        chat_model=row.chat_model,
        mini_model=row.mini_model,
        embedding_provider=row.embedding_provider,
        embedding_api_url=row.embedding_api_url,
        embedding_token=row.embedding_token,
        embedding_model=row.embedding_model,
    )
    try:
        return validate_ai_settings(settings)
    except ValueError as exc:
        raise GuildAIConfigError(
            f"Некорректная AI-конфигурация гильдии {row.guild_id}: {exc}"
        ) from None


def _settings_values(settings: AISettings) -> tuple[str | None, ...]:
    """Возвращает внутреннюю последовательность для сравнения двух копий."""
    return (
        settings.provider,
        settings.api_url,
        settings.token,
        settings.chat_model,
        settings.mini_model,
        settings.embedding_provider,
        settings.embedding_api_url,
        settings.embedding_token,
        settings.embedding_model,
    )


async def get(session: DualSessionProxy, guild_id: int) -> AISettings | None:
    """Читает и валидирует одинаковую конфигурацию из обеих БД."""
    local = await _read_side(session.local, guild_id)
    remote = await _read_side(session.remote, guild_id)
    if local is None and remote is None:
        return None
    if local is None or remote is None or _settings_values(local) != _settings_values(remote):
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не согласована в базах данных")
    return local


async def load_all(session: DualSessionProxy) -> dict[int, AISettings]:
    """Загружает все сохранённые строки без env fallback."""
    result = await session.execute(select(GuildAIConfig))
    settings_by_guild: dict[int, AISettings] = {}
    for row in result.scalars().all():
        settings_by_guild[row.guild_id] = _row_to_settings(row)
    return settings_by_guild


def _coerce_environment_settings(
    environment_settings: AISettings | Mapping[str, str | None] | None,
) -> AISettings:
    """Принимает настройки env в типизированном или тестовом mapping-виде."""
    if environment_settings is None:
        return get_ai_settings()
    if isinstance(environment_settings, AISettings):
        return environment_settings
    defaults = get_ai_settings()

    def value(name: str, env_name: str) -> str | None:
        """Читает canonical field или его environment alias."""
        return environment_settings.get(name) or environment_settings.get(env_name)

    return AISettings(
        provider=value("provider", "AI_PROVIDER") or defaults.provider,
        api_url=value("api_url", "AI_API_URL") or defaults.api_url,
        token=value("token", "AI_TOKEN"),
        chat_model=value("chat_model", "AI_MODEL") or defaults.chat_model,
        mini_model=value("mini_model", "AI_MODEL_MINI") or defaults.mini_model,
        embedding_provider=(
            value("embedding_provider", "AI_EMBEDDING_PROVIDER") or defaults.embedding_provider
        ),
        embedding_api_url=(
            value("embedding_api_url", "AI_EMBEDDING_API_URL") or defaults.embedding_api_url
        ),
        embedding_token=value("embedding_token", "AI_EMBEDDING_TOKEN"),
        embedding_model=(value("embedding_model", "AI_EMBEDDING_MODEL") or defaults.embedding_model),
    )


async def _read_side(session: object, guild_id: int) -> AISettings | None:
    """Читает строку из конкретной SQLAlchemy-сессии."""
    result = await session.execute(select(GuildAIConfig).where(GuildAIConfig.guild_id == guild_id))
    row = result.scalar_one_or_none()
    return None if row is None else _row_to_settings(row)


async def _read_committed_pair(session: DualSessionProxy, guild_id: int) -> AISettings:
    """Проверяет наличие одинаковой строки в обеих БД после commit."""
    local = await _read_side(session.local, guild_id)
    remote = await _read_side(session.remote, guild_id)
    if local is None or remote is None or _settings_values(local) != _settings_values(remote):
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не согласована в базах данных")
    return local


async def _ensure_unlocked(
    session: DualSessionProxy,
    guild_id: int,
    environment_settings: AISettings | Mapping[str, str | None] | None,
) -> AISettings:
    """Выполняет ensure без захвата per-guild lock."""
    existing = await get(session, guild_id)
    if existing is not None:
        return existing

    try:
        candidate = _coerce_environment_settings(environment_settings)
        candidate = validate_ai_settings(candidate)
    except ValueError as exc:
        raise GuildAIConfigError(
            f"Нельзя создать AI-конфигурацию гильдии {guild_id}: {exc}"
        ) from None

    row = GuildAIConfig(
        guild_id=guild_id,
        provider=candidate.provider,
        api_url=candidate.api_url,
        token=candidate.token,
        chat_model=candidate.chat_model,
        mini_model=candidate.mini_model,
        embedding_provider=candidate.embedding_provider,
        embedding_api_url=candidate.embedding_api_url,
        embedding_token=candidate.embedding_token,
        embedding_model=candidate.embedding_model,
    )
    try:
        session.add(row)
        await session.commit(atomic=True)
    except IntegrityError:
        await session.rollback()
        winner = await get(session, guild_id)
        if winner is not None:
            return winner
        raise GuildAIConfigError(
            f"Не удалось сохранить AI-конфигурацию гильдии {guild_id}"
        ) from None
    except Exception:
        await session.rollback()
        raise GuildAIConfigError(
            f"Не удалось сохранить AI-конфигурацию гильдии {guild_id}"
        ) from None

    return await _read_committed_pair(session, guild_id)


async def ensure(
    session: DualSessionProxy,
    guild_id: int,
    environment_settings: AISettings | Mapping[str, str | None] | None = None,
) -> AISettings:
    """Возвращает существующую строку или атомарно создаёт полную новую."""
    async with _lock_for(guild_id):
        return await _ensure_unlocked(session, guild_id, environment_settings)


def clear_locks() -> None:
    """Очищает неактивные locks после сброса приложения."""
    for guild_id, lock in list(_locks.items()):
        if not lock.locked():
            _locks.pop(guild_id, None)


async def update_model(
    session: DualSessionProxy,
    guild_id: int,
    field: str,
    model_name: str,
) -> None:
    """Обновляет chat_model или mini_model для гильдии.

    ``field`` должен быть ``'chat_model'`` или ``'mini_model'``.
    Выбрасывает ``GuildAIConfigError`` если запись не найдена или field некорректен.
    """
    if field not in {"chat_model", "mini_model"}:
        raise GuildAIConfigError(f"Некорректное поле модели: {field!r}")

    from sqlalchemy import update as sa_update

    stmt = (
        sa_update(GuildAIConfig)
        .where(GuildAIConfig.guild_id == guild_id)
        .values(**{field: model_name})
    )
    stmt.is_dml = True  # type: ignore[attr-defined]
    result = await session.execute(stmt)
    if result.rowcount == 0:
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не найдена")
    await session.commit()
    from app.data import guild_ai_config_cache
    guild_ai_config_cache.invalidate(guild_id)
    await guild_ai_config_cache.ensure(session, guild_id)


async def update_provider(
    session: DualSessionProxy,
    guild_id: int,
    provider_type: str,
    name: str,
    api_url: str,
    token: str,
) -> None:
    """Обновляет провайдера (имя, адрес, токен) для гильдии.

    ``provider_type`` должен быть ``'chat'`` или ``'embedding'``.
    Выбрасывает ``GuildAIConfigError`` если запись не найдена или provider_type некорректен.
    """
    if provider_type not in {"chat", "embedding"}:
        raise GuildAIConfigError(f"Некорректный тип провайдера: {provider_type!r}")

    if provider_type == "chat":
        values = {"provider": name, "api_url": api_url, "token": token}
    else:
        values = {
            "embedding_provider": name,
            "embedding_api_url": api_url,
            "embedding_token": token,
        }

    from sqlalchemy import update as sa_update

    stmt = (
        sa_update(GuildAIConfig)
        .where(GuildAIConfig.guild_id == guild_id)
        .values(**values)
    )
    stmt.is_dml = True  # type: ignore[attr-defined]
    result = await session.execute(stmt)
    if result.rowcount == 0:
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не найдена")
    await session.commit()
    from app.data import guild_ai_config_cache
    guild_ai_config_cache.invalidate(guild_id)
    await guild_ai_config_cache.ensure(session, guild_id)

