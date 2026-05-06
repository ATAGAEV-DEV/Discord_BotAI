from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import UserDescription, async_session

# Глобальный кэш: {guild_id: {nick: description}}
_cache: dict[int, dict[str, str]] = {}
_loaded: bool = False


async def load_all() -> None:
    """Загружает все описания из БД в кэш.

    Вызывается один раз при старте бота (on_ready).
    """
    global _loaded  # noqa: PLW0603
    async with async_session() as session:
        result = await session.execute(select(UserDescription))
        rows = result.scalars().all()

    _cache.clear()
    for row in rows:
        _cache.setdefault(row.guild_id, {})[row.nick] = row.description

    _loaded = True
    print(f"Кэш описаний загружен: {sum(len(v) for v in _cache.values())} записей")


def get(guild_id: int) -> dict[str, str]:
    """Возвращает описания пользователей для сервера из кэша.

    Мгновенный доступ без запросов к БД.
    """
    return dict(_cache.get(guild_id, {}))


def get_all() -> dict[str, str]:
    """Возвращает объединённый словарь описаний со всех серверов.

    Используется как прямая замена старого USER_DESCRIPTIONS.
    При конфликте ников — побеждает последний по guild_id.
    """
    merged: dict[str, str] = {}
    for guild_descs in _cache.values():
        merged.update(guild_descs)
    return merged


async def save(nick: str, description: str, guild_id: int) -> str:
    """Сохраняет описание в БД и обновляет кэш."""
    async with async_session() as session:
        existing = await _find_existing(session, nick, guild_id)

        if existing:
            existing.description = description
            action = "обновлено"
        else:
            session.add(UserDescription(nick=nick, description=description, guild_id=guild_id))
            action = "добавлено"

        await session.commit()

    # Обновляем кэш
    _cache.setdefault(guild_id, {})[nick] = description
    return f"Описание для '{nick}' успешно {action}!"


async def remove(nick: str, guild_id: int) -> str:
    """Удаляет описание из БД и кэша."""
    async with async_session() as session:
        query = sa_delete(UserDescription).where(
            UserDescription.nick == nick, UserDescription.guild_id == guild_id
        )
        result = await session.execute(query)
        await session.commit()

    # Обновляем кэш
    guild_cache = _cache.get(guild_id, {})
    removed = guild_cache.pop(nick, None)

    if result.rowcount > 0 or removed is not None:
        return f"Описание для '{nick}' удалено."
    return f"Описание для '{nick}' не найдено."


async def _find_existing(
    session: AsyncSession, nick: str, guild_id: int
) -> UserDescription | None:
    """Находит существующую запись описания пользователя."""
    query = select(UserDescription).where(
        UserDescription.nick == nick, UserDescription.guild_id == guild_id
    )
    result = await session.execute(query)
    return result.scalar_one_or_none()
