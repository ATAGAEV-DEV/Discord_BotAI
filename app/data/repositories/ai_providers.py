"""Repository для справочника AI-провайдеров."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.data.models import AIProvider

if TYPE_CHECKING:
    from app.data.models import DualSessionProxy


async def get_all(session: DualSessionProxy) -> list[tuple[str, str]]:
    """Возвращает все записи (имя, api_url) в алфавитном порядке."""
    result = await session.execute(select(AIProvider).order_by(AIProvider.name))
    return [(row.name, row.api_url) for row in result.scalars().all()]


async def add(session: DualSessionProxy, name: str, api_url: str, token: str) -> str:
    """Добавляет провайдера в справочник.

    Возвращает сообщение об успехе или о том, что провайдер уже существует.
    """
    name = name.strip()
    api_url = api_url.strip()
    token = token.strip()
    if not name:
        return "❌ Имя провайдера не может быть пустым."
    if not api_url:
        return "❌ Адрес провайдера не может быть пустым."
    if not token:
        return "❌ Токен провайдера не может быть пустым."
    row = AIProvider(name=name, api_url=api_url, token=token)
    try:
        session.add(row)
        await session.commit()
        return f"✅ Провайдер `{name}` добавлен."
    except IntegrityError:
        await session.rollback()
        return f"⚠️ Провайдер `{name}` уже есть в списке."
    except Exception:
        await session.rollback()
        raise


async def remove(session: DualSessionProxy, name: str) -> str:
    """Удаляет провайдера из справочника.

    Возвращает сообщение об успехе или о том, что провайдер не найден.
    """
    name = name.strip()
    stmt = delete(AIProvider).where(AIProvider.name == name)
    stmt.is_dml = True  # type: ignore[attr-defined]
    result = await session.execute(stmt)
    await session.commit()
    if result.rowcount == 0:
        return f"⚠️ Провайдер `{name}` не найден в списке."
    return f"✅ Провайдер `{name}` удалён."


async def get(session: DualSessionProxy, name: str) -> AIProvider | None:
    """Возвращает запись провайдера по имени или None."""
    result = await session.execute(
        select(AIProvider).where(AIProvider.name == name.strip())
    )
    return result.scalar_one_or_none()
