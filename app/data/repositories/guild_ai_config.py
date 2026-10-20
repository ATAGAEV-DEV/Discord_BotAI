"""Repository полной AI-конфигурации конкретной Discord-гильдии."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AISettings, get_ai_settings, validate_ai_settings
from app.data.models import GuildAIConfig


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
    except (TypeError, ValueError) as exc:
        raise GuildAIConfigError(
            f"Некорректная AI-конфигурация гильдии {row.guild_id}: {exc}"
        ) from None


async def get(session: AsyncSession, guild_id: int) -> AISettings | None:
    """Читает и валидирует конфигурацию гильдии из БД."""
    result = await session.execute(
        select(GuildAIConfig).where(GuildAIConfig.guild_id == guild_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return _row_to_settings(row)


async def load_all(session: AsyncSession) -> dict[int, AISettings]:
    """Загружает все сохранённые строки без env fallback."""
    result = await session.execute(select(GuildAIConfig))
    settings_by_guild: dict[int, AISettings] = {}
    for row in result.scalars().all():
        try:
            settings_by_guild[row.guild_id] = _row_to_settings(row)
        except GuildAIConfigError:
            pass
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

    provider = value("provider", "AI_PROVIDER") or defaults.provider
    api_url = value("api_url", "AI_API_URL") or defaults.api_url

    return AISettings(
        provider=provider,
        api_url=api_url,
        token=value("token", "AI_TOKEN"),
        chat_model=value("chat_model", "AI_MODEL"),
        mini_model=value("mini_model", "AI_MODEL_MINI"),
        embedding_provider=value("embedding_provider", "AI_EMBEDDING_PROVIDER") or provider,
        embedding_api_url=value("embedding_api_url", "AI_EMBEDDING_API_URL") or api_url,
        embedding_token=value("embedding_token", "AI_EMBEDDING_TOKEN"),
        embedding_model=value("embedding_model", "AI_EMBEDDING_MODEL"),
    )


async def _ensure_unlocked(
    session: AsyncSession,
    guild_id: int,
    environment_settings: AISettings | Mapping[str, str | None] | None,
) -> AISettings:
    """Возвращает существующую строку или атомарно создаёт полную новую (без lock)."""
    existing = await get(session, guild_id)
    if existing is not None:
        return existing

    env = _coerce_environment_settings(environment_settings)
    try:
        env = validate_ai_settings(env)
    except (TypeError, ValueError) as exc:
        raise GuildAIConfigError(
            f"Невозможно создать AI-конфигурацию гильдии {guild_id}: "
            f"{exc}"
        ) from None

    new_row = GuildAIConfig(
        guild_id=guild_id,
        provider=env.provider,
        api_url=env.api_url,
        token=env.token,
        chat_model=env.chat_model,
        mini_model=env.mini_model,
        embedding_provider=env.embedding_provider,
        embedding_api_url=env.embedding_api_url,
        embedding_token=env.embedding_token,
        embedding_model=env.embedding_model,
    )
    session.add(new_row)
    try:
        await session.commit()
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

    result = await get(session, guild_id)
    if result is None:
        raise GuildAIConfigError(
            f"Не удалось сохранить AI-конфигурацию гильдии {guild_id}"
        )
    return result


async def ensure(
    session: AsyncSession,
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
    session: AsyncSession,
    guild_id: int,
    field: str,
    model_name: str,
) -> None:
    """Обновляет chat_model или mini_model для гильдии.

    ``field`` должен быть ``'chat_model'``, ``'mini_model'`` или
    ``'embedding_model'``.
    Выбрасывает ``GuildAIConfigError`` если запись не найдена или field некорректен.
    """
    if field not in {"chat_model", "mini_model", "embedding_model"}:
        raise GuildAIConfigError(f"Некорректное поле модели: {field!r}")
    if not isinstance(model_name, str) or not model_name.strip():
        raise GuildAIConfigError(f"Модель {field} не задана")

    from sqlalchemy import update as sa_update

    stmt = (
        sa_update(GuildAIConfig)
        .where(GuildAIConfig.guild_id == guild_id)
        .values(**{field: model_name})
    )
    result = await session.execute(stmt)
    if result.rowcount == 0:
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не найдена")
    await session.commit()
    from app.data import guild_ai_config_cache
    guild_ai_config_cache.invalidate(guild_id)
    await guild_ai_config_cache.ensure(session, guild_id)


async def update_provider(
    session: AsyncSession,
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
    for field_name, value in (("provider", name), ("api_url", api_url), ("token", token)):
        if not isinstance(value, str) or not value.strip():
            raise GuildAIConfigError(f"Поле {field_name} не задано")

    current = await get(session, guild_id)
    if current is None:
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не найдена")

    name = name.strip()
    api_url = api_url.strip()
    token = token.strip()
    candidate = AISettings(
        provider=name if provider_type == "chat" else current.provider,
        api_url=api_url if provider_type == "chat" else current.api_url,
        token=token if provider_type == "chat" else current.token,
        chat_model=current.chat_model,
        mini_model=current.mini_model,
        embedding_provider=name if provider_type == "embedding" else current.embedding_provider,
        embedding_api_url=api_url if provider_type == "embedding" else current.embedding_api_url,
        embedding_token=token if provider_type == "embedding" else current.embedding_token,
        embedding_model=current.embedding_model,
    )
    try:
        validate_ai_settings(candidate)
    except (TypeError, ValueError) as exc:
        raise GuildAIConfigError(
            f"Невозможно обновить AI-конфигурацию гильдии {guild_id}: {exc}"
        ) from None

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
    result = await session.execute(stmt)
    if result.rowcount == 0:
        raise GuildAIConfigError(f"AI-конфигурация гильдии {guild_id} не найдена")
    await session.commit()
    from app.data import guild_ai_config_cache
    guild_ai_config_cache.invalidate(guild_id)
    await guild_ai_config_cache.ensure(session, guild_id)

