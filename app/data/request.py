from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.decorators import db_operation
from app.data.models import (
    ChannelMessage,
    UserDescription,
    UserMessageStats,
)
from app.tools.utils import get_rank_description

__all__ = ["db_operation"]


@db_operation("сохранении сообщения канала")
async def save_channel_message(
    session: AsyncSession, channel_id: int, message_id: int, author: str, content: str
) -> None:
    """Сохраняет сообщение канала в базы данных."""
    new_message = ChannelMessage(
        channel_id=channel_id, message_id=message_id, author=author, content=content
    )
    session.add(new_message)
    await session.commit()


@db_operation("получении сообщений канала")
async def get_channel_messages(session: AsyncSession, channel_id: int) -> list[ChannelMessage]:
    """Извлекает сообщения канала из базы данных."""
    query = select(ChannelMessage).where(ChannelMessage.channel_id == channel_id)
    result = await session.execute(query)
    return result.scalars().all()


@db_operation("удалении сообщений канала")
async def delete_channel_messages(session: AsyncSession, channel_id: int) -> None:
    """Удаляет сообщения канала из базы данных."""
    query = delete(ChannelMessage).where(ChannelMessage.channel_id == channel_id)
    await session.execute(query)
    await session.commit()


@db_operation("обновлении статистики сообщений")
async def update_message_count(
    session: AsyncSession, user_id: int, name: str, guild_id: int
) -> dict:
    """Обновляет счетчик сообщений пользователя и возвращает информацию о повышении ранга."""
    query = select(UserMessageStats).where(
        UserMessageStats.user_id == user_id, UserMessageStats.guild_id == guild_id
    )
    result = await session.execute(query)
    user_stats = result.scalar_one_or_none()

    if user_stats:
        old_count = user_stats.message_count
        old_rank = get_rank_description(old_count)
        new_count = old_count + 1
        new_rank = get_rank_description(new_count)

        stmt = (
            update(UserMessageStats)
            .where(UserMessageStats.user_id == user_id, UserMessageStats.guild_id == guild_id)
            .values(message_count=new_count, last_updated=func.now())
        )
        await session.execute(stmt)

        rank_up = new_rank["rank_level"] > old_rank["rank_level"]
    else:
        old_rank = get_rank_description(0)
        new_count = 1
        new_rank = get_rank_description(new_count)

        new_stat = UserMessageStats(
            user_id=user_id, name=name, guild_id=guild_id, message_count=new_count
        )
        session.add(new_stat)

        rank_up = new_rank["rank_level"] > old_rank["rank_level"]

    await session.commit()

    return {
        "rank_up": rank_up,
        "old_rank": old_rank["rank_level"],
        "new_rank": new_rank["rank_level"],
        "message_count": new_count,
    }


@db_operation("получении статистики сообщений")
async def get_rank(session: AsyncSession, user_id: int, guild_id: int) -> int:
    """Получает количество сообщений пользователя на сервере."""
    query = select(UserMessageStats.message_count).where(
        UserMessageStats.user_id == user_id, UserMessageStats.guild_id == guild_id
    )
    result = await session.execute(query)
    stats = result.scalar_one_or_none()

    count = stats if stats is not None else 0
    return int(count)


@db_operation("получении ранга пользователя")
async def get_user_rank(session: AsyncSession, user_id: int, guild_id: int) -> int:
    """Получает ранг пользователя в указанном сервере на основе количества сообщений."""
    subquery = (
        select(
            UserMessageStats.user_id,
            func.rank()
            .over(
                partition_by=UserMessageStats.guild_id,
                order_by=UserMessageStats.message_count.desc(),
            )
            .label("user_rank"),
        )
        .where(UserMessageStats.guild_id == guild_id)
        .subquery()
    )

    query = select(subquery.c.user_rank).where(and_(subquery.c.user_id == user_id))

    result = await session.execute(query)
    user_rank = result.scalar_one_or_none()

    return user_rank if user_rank is not None else 0


@db_operation("получении описаний пользователей")
async def get_user_descriptions(session: AsyncSession, guild_id: int) -> dict[str, str]:
    """Получает описания пользователей для указанного сервера.

    Возвращает словарь {nick: description}, аналогичный старому USER_DESCRIPTIONS.
    """
    query = select(UserDescription).where(UserDescription.guild_id == guild_id)
    result = await session.execute(query)
    rows = result.scalars().all()
    return {row.nick: row.description for row in rows}


@db_operation("сохранении описания пользователя")
async def save_user_description(
    session: AsyncSession, nick: str, description: str, guild_id: int
) -> str:
    """Сохраняет или обновляет описание пользователя для сервера."""
    query = select(UserDescription).where(
        UserDescription.nick == nick, UserDescription.guild_id == guild_id
    )
    result = await session.execute(query)
    existing = result.scalar_one_or_none()

    if existing:
        existing.description = description
        action = "обновлено"
    else:
        new_entry = UserDescription(nick=nick, description=description, guild_id=guild_id)
        session.add(new_entry)
        action = "добавлено"

    await session.commit()
    return f"Описание для '{nick}' успешно {action}!"


@db_operation("удалении описания пользователя")
async def delete_user_description(session: AsyncSession, nick: str, guild_id: int) -> str:
    """Удаляет описание пользователя для указанного сервера."""
    query = delete(UserDescription).where(
        UserDescription.nick == nick, UserDescription.guild_id == guild_id
    )
    result = await session.execute(query)
    await session.commit()

    if result.rowcount > 0:
        return f"Описание для '{nick}' удалено."
    return f"Описание для '{nick}' не найдено."
