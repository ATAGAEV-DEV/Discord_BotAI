"""Постоянное хранение завершённых диалоговых пар каналов."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import CONVERSATION_MAX_PAIRS
from app.data.decorators import db_operation
from app.data.models import ConversationPair


@db_operation("сохранении пары диалога")
async def save_pair(
    session: AsyncSession,
    guild_id: int,
    channel_id: int,
    user_content: str,
    assistant_content: str,
) -> None:
    """Атомарно сохраняет вопрос и ответ без преобразования их содержимого."""
    session.add(
        ConversationPair(
            guild_id=guild_id,
            channel_id=channel_id,
            user_content=user_content,
            assistant_content=assistant_content,
        )
    )
    await session.commit()


@db_operation("восстановлении истории каналов")
async def load_recent(
    session: AsyncSession,
    guild_ids: list[int],
    max_pairs: int = CONVERSATION_MAX_PAIRS,
) -> list[ConversationPair]:
    """Выбирает последние пары каждого канала в порядке их сохранения."""
    if not guild_ids or max_pairs <= 0:
        return []

    ranked = (
        select(
            ConversationPair.id,
            func.row_number()
            .over(
                partition_by=(ConversationPair.guild_id, ConversationPair.channel_id),
                order_by=ConversationPair.id.desc(),
            )
            .label("position"),
        )
        .where(ConversationPair.guild_id.in_(guild_ids))
        .subquery()
    )
    query = (
        select(ConversationPair)
        .join(ranked, ConversationPair.id == ranked.c.id)
        .where(ranked.c.position <= max_pairs)
        .order_by(ConversationPair.guild_id, ConversationPair.channel_id, ConversationPair.id)
    )
    result = await session.execute(query)
    return list(result.scalars().all())
