"""Unit-тесты репозитория ai_models."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.data.models import AIModel
from app.data.repositories import ai_models

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(rows: list[AIModel] | None = None, *, rowcount: int = 1) -> MagicMock:
    """Возвращает мок AsyncSession."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    scalars_ns = SimpleNamespace(all=lambda: rows or [])
    result_ns = SimpleNamespace(
        scalars=lambda: scalars_ns,
        scalar_one_or_none=lambda: (rows[0] if rows else None),
        rowcount=rowcount,
    )
    session.execute = AsyncMock(return_value=result_ns)
    return session


def _model(name: str) -> AIModel:
    row = AIModel()
    row.name = name
    return row


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_returns_names() -> None:
    """get_all возвращает список имён моделей."""
    session = _make_session([_model("model-a"), _model("model-b")])
    result = await ai_models.get_all(session)
    assert result == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_get_all_empty() -> None:
    """get_all возвращает пустой список если таблица пуста."""
    session = _make_session([])
    result = await ai_models.get_all(session)
    assert result == []


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_success() -> None:
    """Add возвращает сообщение об успехе и делает commit."""
    session = _make_session()
    msg = await ai_models.add(session, "new-model")
    assert "new-model" in msg
    assert "✅" in msg
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_empty_name() -> None:
    """Add отклоняет пустое имя без обращения к БД."""
    session = _make_session()
    msg = await ai_models.add(session, "   ")
    assert "❌" in msg
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_duplicate_returns_warning() -> None:
    """Add возвращает предупреждение при IntegrityError и делает rollback."""
    from sqlalchemy.exc import IntegrityError

    session = _make_session()
    session.commit = AsyncMock(side_effect=IntegrityError("", {}, Exception()))
    msg = await ai_models.add(session, "dup-model")
    assert "⚠️" in msg
    assert "dup-model" in msg
    session.rollback.assert_awaited_once()


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_existing() -> None:
    """Remove возвращает сообщение об успехе когда rowcount=1."""
    session = _make_session(rowcount=1)
    msg = await ai_models.remove(session, "old-model")
    assert "✅" in msg
    assert "old-model" in msg
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_not_found() -> None:
    """Remove возвращает предупреждение когда rowcount=0."""
    session = _make_session(rowcount=0)
    msg = await ai_models.remove(session, "ghost-model")
    assert "⚠️" in msg
    assert "ghost-model" in msg


@pytest.mark.asyncio
async def test_add_reraises_unexpected_exception_with_rollback() -> None:
    """Add делает rollback и повторно выбрасывает неожиданное исключение."""
    session = _make_session()
    session.commit = AsyncMock(side_effect=RuntimeError("unexpected db error"))

    with pytest.raises(RuntimeError, match="unexpected db error"):
        await ai_models.add(session, "broken-model")

    session.rollback.assert_awaited_once()


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exists_true() -> None:
    """Exists возвращает True когда модель найдена."""
    session = _make_session([_model("some-model")])
    assert await ai_models.exists(session, "some-model") is True


@pytest.mark.asyncio
async def test_exists_false() -> None:
    """Exists возвращает False когда модель не найдена."""
    session = _make_session([])
    assert await ai_models.exists(session, "missing") is False

