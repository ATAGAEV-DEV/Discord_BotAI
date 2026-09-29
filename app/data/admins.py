from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.decorators import db_operation
from app.data.models import BotAdmin

# Кэш администраторов: {guild_id: {user_id: username}}
_cache: dict[int, dict[int, str]] = {}


@db_operation("загрузке администраторов бота")
async def load_all(session: AsyncSession) -> None:
    """Загружает администраторов всех серверов в кэш."""
    result = await session.execute(select(BotAdmin))
    rows = result.scalars().all()

    _cache.clear()
    for row in rows:
        _cache.setdefault(row.guild_id, {})[row.user_id] = row.username

    print(f"Кэш администраторов загружен: {sum(len(v) for v in _cache.values())} записей")


def is_admin(guild_id: int, user_id: int) -> bool:
    """Проверяет наличие пользователя среди администраторов сервера."""
    return user_id in _cache.get(guild_id, {})


def get_all(guild_id: int) -> dict[int, str]:
    """Возвращает администраторов указанного сервера из кэша."""
    return dict(_cache.get(guild_id, {}))


@db_operation("добавлении администратора бота")
async def add(session: AsyncSession, guild_id: int, user_id: int, username: str) -> str:
    """Добавляет администратора или обновляет его отображаемое имя."""
    query = select(BotAdmin).where(
        BotAdmin.guild_id == guild_id,
        BotAdmin.user_id == user_id,
    )
    result = await session.execute(query)
    existing = result.scalar_one_or_none()

    if existing is not None:
        await session.execute(
            update(BotAdmin)
            .where(BotAdmin.guild_id == guild_id, BotAdmin.user_id == user_id)
            .values(username=username)
        )
        action = "имя администратора обновлено"
    else:
        session.add(BotAdmin(guild_id=guild_id, user_id=user_id, username=username))
        action = "администратор добавлен"

    await session.commit()
    _cache.setdefault(guild_id, {})[user_id] = username
    return f"✅ {action}: **{username}** ({user_id})"


@db_operation("удалении администратора бота")
async def remove(session: AsyncSession, guild_id: int, user_id: int) -> str:
    """Удаляет администратора текущего сервера."""
    query = delete(BotAdmin).where(
        BotAdmin.guild_id == guild_id,
        BotAdmin.user_id == user_id,
    )
    result = await session.execute(query)
    await session.commit()

    removed = _cache.get(guild_id, {}).pop(user_id, None)
    if result.rowcount > 0 or removed is not None:
        return f"✅ Администратор **{user_id}** удалён."
    return f"ℹ️ Администратор **{user_id}** не найден."
