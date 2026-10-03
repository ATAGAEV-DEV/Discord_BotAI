from sqlalchemy import delete as sa_delete
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.decorators import db_operation
from app.data.models import GuildEmoji

_cache: dict[int, dict[str, str]] = {}


@db_operation("загрузке описаний эмодзи")
async def load_all(session: AsyncSession) -> None:
    """Загружает описания эмодзи всех серверов в кэш."""
    result = await session.execute(select(GuildEmoji))
    rows = result.scalars().all()

    _cache.clear()
    for row in rows:
        _cache.setdefault(row.guild_id, {})[row.name] = row.description


def get(guild_id: int) -> dict[str, str]:
    """Возвращает копию описаний эмодзи указанного сервера."""
    return dict(_cache.get(guild_id, {}))


@db_operation("сохранении описания эмодзи")
async def save(session: AsyncSession, name: str, description: str, guild_id: int) -> str:
    """Сохраняет или обновляет описание эмодзи в БД и кэше."""
    query = select(GuildEmoji).where(
        GuildEmoji.name == name,
        GuildEmoji.guild_id == guild_id,
    )
    result = await session.execute(query)
    existing = result.scalar_one_or_none()

    if existing:
        await session.execute(
            update(GuildEmoji)
            .where(GuildEmoji.name == name, GuildEmoji.guild_id == guild_id)
            .values(description=description)
        )
        action = "обновлено"
    else:
        session.add(GuildEmoji(name=name, description=description, guild_id=guild_id))
        action = "добавлено"

    await session.commit()
    _cache.setdefault(guild_id, {})[name] = description
    return f"Описание эмодзи '{name}' успешно {action}!"


@db_operation("удалении описания эмодзи")
async def remove(session: AsyncSession, name: str, guild_id: int) -> str:
    """Удаляет описание эмодзи из БД и кэша."""
    query = sa_delete(GuildEmoji).where(
        GuildEmoji.name == name,
        GuildEmoji.guild_id == guild_id,
    )
    result = await session.execute(query)
    await session.commit()

    removed = _cache.get(guild_id, {}).pop(name, None)
    if result.rowcount > 0 or removed is not None:
        return f"Описание эмодзи '{name}' удалено."
    return f"Описание эмодзи '{name}' не найдено."
