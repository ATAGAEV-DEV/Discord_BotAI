import asyncio
from collections.abc import Callable
from datetime import date, datetime
from functools import wraps
from typing import Any

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import (
    Birthday,
    ChannelMessage,
    Holiday,
    UserDescription,
    UserMessageStats,
    async_session,
)
from app.tools.utils import get_rank_description

DB_TIMEOUT = 10


def db_operation(operation_name: str) -> Callable:
    """Декоратор для устранения бойлерплейта БД-операций.

    Оборачивает функцию в async with session + try/except с таймаутом.
    Декорируемая функция должна принимать session: AsyncSession первым аргументом.
    """

    def decorator(func_inner: Callable) -> Callable:
        @wraps(func_inner)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with async_session() as session:
                try:
                    return await asyncio.wait_for(
                        func_inner(session, *args, **kwargs), timeout=DB_TIMEOUT
                    )
                except TimeoutError:
                    raise Exception(f"Таймаут при {operation_name}.")
                except Exception as e:
                    if isinstance(e, (ValueError, Exception)) and "Таймаут" in str(e):
                        raise
                    raise Exception(f"Ошибка при {operation_name}: {e}")

        return wrapper

    return decorator


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


@db_operation("сохранении даты рождения")
async def save_birthday(
    session: AsyncSession, user_id: int, display_name: str, name: str, birthday: datetime
) -> None:
    """Сохраняет дату рождения в БД."""
    query = select(Birthday).where(Birthday.user_id == user_id)
    result = await session.execute(query)
    birthday_entry = result.scalar()
    if birthday_entry:
        birthday_entry.birthday = birthday
        birthday_entry.display_name = display_name
        birthday_entry.name = name
    else:
        birthday_entry = Birthday(
            user_id=user_id, display_name=display_name, name=name, birthday=birthday
        )
        session.add(birthday_entry)
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


@db_operation("проверке праздника")
async def check_holiday(session: AsyncSession, current_date: date) -> str | None:
    """Проверяет, является ли текущая дата праздником."""
    query = select(Holiday).where(
        Holiday.day == current_date.day, Holiday.month == current_date.month
    )
    result = await session.execute(query)
    holiday = result.scalar_one_or_none()
    return holiday.name if holiday else None


@db_operation("сохранении праздника")
async def save_holiday(session: AsyncSession, day: int, month: int, holiday_name: str) -> str:
    """Сохраняет праздник в БД."""
    query = select(Holiday).where(Holiday.day == day, Holiday.month == month)
    result = await session.execute(query)
    existing_holiday = result.scalar_one_or_none()

    if existing_holiday:
        existing_holiday.name = holiday_name
        action = "обновлен"
    else:
        new_holiday = Holiday(day=day, month=month, name=holiday_name)
        session.add(new_holiday)
        action = "добавлен"

    await session.commit()
    date_str = f"{day:02d}.{month:02d}"
    return f"Праздник '{holiday_name}' на {date_str} успешно {action}!"


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

