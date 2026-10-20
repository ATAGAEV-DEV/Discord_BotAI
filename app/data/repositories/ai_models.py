"""Repository для справочника AI-моделей."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.models import AIModel


async def get_all(session: AsyncSession) -> list[str]:
    """Возвращает все имена моделей в алфавитном порядке."""
    result = await session.execute(select(AIModel).order_by(AIModel.name))
    return [row.name for row in result.scalars().all()]


async def add(session: AsyncSession, name: str) -> str:
    """Добавляет модель в справочник.

    Возвращает сообщение об успехе или о том, что модель уже существует.
    """
    name = name.strip()
    if not name:
        return "❌ Имя модели не может быть пустым."
    row = AIModel(name=name)
    try:
        session.add(row)
        await session.commit()
        return f"✅ Модель `{name}` добавлена."
    except IntegrityError:
        await session.rollback()
        return f"⚠️ Модель `{name}` уже есть в списке."
    except Exception:
        await session.rollback()
        raise


async def remove(session: AsyncSession, name: str) -> str:
    """Удаляет модель из справочника.

    Возвращает сообщение об успехе или о том, что модель не найдена.
    """
    name = name.strip()
    stmt = delete(AIModel).where(AIModel.name == name)
    result = await session.execute(stmt)
    await session.commit()
    if result.rowcount == 0:
        return f"⚠️ Модель `{name}` не найдена в списке."
    return f"✅ Модель `{name}` удалена."


async def exists(session: AsyncSession, name: str) -> bool:
    """Возвращает True если модель с таким именем есть в справочнике."""
    result = await session.execute(select(AIModel).where(AIModel.name == name.strip()))
    return result.scalar_one_or_none() is not None

