"""Unit-тесты repository AI-провайдеров."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.data.models import AIProvider
from app.data.repositories import ai_providers


class _Result:
    """Минимальный fake SQLAlchemy result для repository тестов."""

    def __init__(self, rows: list[AIProvider]) -> None:
        self._rows = rows

    def scalars(self) -> SimpleNamespace:
        """Возвращает fake scalar result."""
        return SimpleNamespace(all=lambda: self._rows)

    def scalar_one_or_none(self) -> "AIProvider | None":
        """Возвращает первую строку или None."""
        return self._rows[0] if self._rows else None

    @property
    def rowcount(self) -> int:
        """Число строк, затронутых операцией."""
        return len(self._rows)


class _Session:
    """Fake AsyncSession для ai_providers repository."""

    def __init__(self, rows: "list[AIProvider] | None" = None) -> None:
        self._rows: list[AIProvider] = rows or []
        self._pending: AIProvider | None = None
        self._deleted_name: str | None = None
        self.commit = AsyncMock(side_effect=self._do_commit)
        self.rollback = AsyncMock()

    async def execute(self, statement: Any) -> _Result:
        """Возвращает текущие строки или результат удаления."""
        stmt_str = str(statement)
        if "DELETE" in stmt_str.upper():
            before = len(self._rows)
            self._rows = [r for r in self._rows if r.name != self._deleted_name]
            deleted_count = before - len(self._rows)
            return _Result([AIProvider(name="x", api_url="u", token="t")] * deleted_count)
        return _Result(list(self._rows))

    def add(self, instance: AIProvider) -> None:
        """Запоминает pending insert."""
        self._pending = instance

    async def _do_commit(self) -> None:
        """Публикует pending insert."""
        if self._pending is not None:
            self._rows.append(self._pending)
            self._pending = None


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_returns_name_and_url_tuples() -> None:
    """get_all возвращает список (name, api_url) для всех провайдеров."""
    rows = [
        AIProvider(name="alpha", api_url="https://alpha.example/v1", token="tok-a"),
        AIProvider(name="beta", api_url="https://beta.example/v1", token="tok-b"),
    ]
    session = _Session(rows)

    result = await ai_providers.get_all(session)

    assert result == [
        ("alpha", "https://alpha.example/v1"),
        ("beta", "https://beta.example/v1"),
    ]


@pytest.mark.asyncio
async def test_get_all_returns_empty_list_when_no_providers() -> None:
    """get_all возвращает [] если провайдеров нет."""
    result = await ai_providers.get_all(_Session([]))
    assert result == []


# ---------------------------------------------------------------------------
# add — validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_rejects_empty_name() -> None:
    """Add возвращает сообщение об ошибке для пустого имени."""
    session = _Session()
    result = await ai_providers.add(session, "  ", "https://p.example/v1", "tok")
    assert "пустым" in result
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_rejects_empty_api_url() -> None:
    """Add возвращает сообщение об ошибке для пустого URL."""
    session = _Session()
    result = await ai_providers.add(session, "my-provider", "  ", "tok")
    assert "пустым" in result
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_rejects_empty_token() -> None:
    """Add возвращает сообщение об ошибке для пустого токена."""
    session = _Session()
    result = await ai_providers.add(session, "my-provider", "https://p.example/v1", "  ")
    assert "пустым" in result
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_inserts_new_provider() -> None:
    """Add сохраняет провайдера и возвращает сообщение об успехе."""
    session = _Session()

    result = await ai_providers.add(session, "my-provider", "https://p.example/v1", "secret")

    assert "✅" in result
    assert "my-provider" in result
    session.commit.assert_awaited_once()
    assert len(session._rows) == 1
    assert session._rows[0].name == "my-provider"


@pytest.mark.asyncio
async def test_add_strips_whitespace() -> None:
    """Add обрезает пробелы вокруг полей перед сохранением."""
    session = _Session()

    await ai_providers.add(session, "  trimmed  ", "  https://p.example/v1  ", "  tok  ")

    assert session._rows[0].name == "trimmed"
    assert session._rows[0].api_url == "https://p.example/v1"


@pytest.mark.asyncio
async def test_add_returns_warning_on_integrity_error() -> None:
    """Add возвращает предупреждение и делает rollback при IntegrityError."""
    session = _Session()
    session.commit = AsyncMock(side_effect=IntegrityError(None, None, Exception()))

    result = await ai_providers.add(session, "dup", "https://p.example/v1", "tok")

    assert "⚠️" in result
    assert "dup" in result
    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_reraises_unexpected_exception() -> None:
    """Add делает rollback и повторно выбрасывает неожиданное исключение."""
    session = _Session()
    session.commit = AsyncMock(side_effect=RuntimeError("db crashed"))

    with pytest.raises(RuntimeError, match="db crashed"):
        await ai_providers.add(session, "p", "https://p.example/v1", "tok")

    session.rollback.assert_awaited_once()


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remove_deletes_existing_provider() -> None:
    """Remove возвращает сообщение об успехе при удалении существующей записи."""
    existing = AIProvider(name="to-remove", api_url="https://r.example/v1", token="t")
    session = _Session([existing])
    session._deleted_name = "to-remove"

    result = await ai_providers.remove(session, "to-remove")

    assert "✅" in result
    assert "to-remove" in result
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_returns_warning_when_not_found() -> None:
    """Remove возвращает предупреждение если провайдер не найден."""
    session = _Session([])
    session._deleted_name = "ghost"

    result = await ai_providers.remove(session, "ghost")

    assert "⚠️" in result
    assert "ghost" in result


@pytest.mark.asyncio
async def test_remove_strips_whitespace() -> None:
    """Remove обрезает пробелы перед поиском записи."""
    existing = AIProvider(name="trimmed", api_url="https://t.example/v1", token="t")
    session = _Session([existing])
    session._deleted_name = "trimmed"

    result = await ai_providers.remove(session, "  trimmed  ")

    assert "✅" in result


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_returns_provider_by_name() -> None:
    """Get возвращает запись провайдера по имени."""
    provider = AIProvider(name="openai", api_url="https://api.openai.com/v1", token="sk-test")
    session = _Session([provider])

    result = await ai_providers.get(session, "openai")

    assert result is provider


@pytest.mark.asyncio
async def test_get_returns_none_when_not_found() -> None:
    """Get возвращает None если провайдер не найден."""
    result = await ai_providers.get(_Session([]), "unknown")
    assert result is None


@pytest.mark.asyncio
async def test_get_strips_whitespace_from_name() -> None:
    """Get обрезает пробелы вокруг имени перед поиском."""
    provider = AIProvider(name="stripped", api_url="https://s.example/v1", token="tok")
    session = _Session([provider])

    result = await ai_providers.get(session, "  stripped  ")

    assert result is provider

