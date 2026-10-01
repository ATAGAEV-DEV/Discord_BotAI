"""Изолированные тесты кэша описаний эмодзи."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.data import decorators, emoji_descriptions_cache
from app.data.models import GuildEmoji


@pytest.fixture(autouse=True)
def empty_emoji_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует глобальный кэш эмодзи между тестами."""
    monkeypatch.setattr(emoji_descriptions_cache, "_cache", {})


@pytest.fixture
def database_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Подменяет сессию декоратора контролируемым mock-объектом."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.context_entered = False
    session.context_exited = False

    @asynccontextmanager
    async def fake_async_session() -> AsyncIterator[MagicMock]:
        session.context_entered = True
        try:
            yield session
        finally:
            session.context_exited = True

    monkeypatch.setattr(decorators, "async_session", fake_async_session)
    return session


def _rows_result(*rows: SimpleNamespace) -> MagicMock:
    """Создаёт результат SQLAlchemy со строками описаний эмодзи."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    return result


def _scalar_result(value: object) -> MagicMock:
    """Создаёт результат SQLAlchemy с найденной или отсутствующей записью."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_load_all_replaces_cache_and_groups_rows(
    database_session: MagicMock,
) -> None:
    """Загрузка очищает старый кэш и группирует описания по серверам."""
    emoji_descriptions_cache._cache[999] = {"old": "stale"}
    database_session.execute.return_value = _rows_result(
        SimpleNamespace(guild_id=100, name="smile", description="Улыбка"),
        SimpleNamespace(guild_id=100, name="heart", description="Сердце"),
        SimpleNamespace(guild_id=200, name="smile", description="Другая улыбка"),
    )

    result = await emoji_descriptions_cache.load_all()

    assert result is None
    database_session.execute.assert_awaited_once()
    assert emoji_descriptions_cache.get(100) == {
        "smile": "Улыбка",
        "heart": "Сердце",
    }
    assert emoji_descriptions_cache.get(200) == {"smile": "Другая улыбка"}
    assert emoji_descriptions_cache.get(999) == {}
    assert database_session.context_entered is True
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_load_all_with_empty_result_clears_cache(
    database_session: MagicMock,
) -> None:
    """Пустой результат загрузки удаляет старые описания."""
    emoji_descriptions_cache._cache[100] = {"smile": "Улыбка"}
    database_session.execute.return_value = _rows_result()

    await emoji_descriptions_cache.load_all()

    assert emoji_descriptions_cache._cache == {}


@pytest.mark.asyncio
async def test_save_updates_existing_row_and_cache(database_session: MagicMock) -> None:
    """Существующая запись обновляется в БД и кэше."""
    existing = SimpleNamespace(description="Старое описание")
    database_session.execute.return_value = _scalar_result(existing)

    message = await emoji_descriptions_cache.save(
        name="smile", description="Новое описание", guild_id=100
    )

    assert message == "Описание эмодзи 'smile' успешно обновлено!"
    assert existing.description == "Новое описание"
    database_session.execute.assert_awaited_once()
    database_session.add.assert_not_called()
    database_session.commit.assert_awaited_once()
    assert emoji_descriptions_cache.get(100) == {"smile": "Новое описание"}


@pytest.mark.asyncio
async def test_save_adds_new_row_and_cache(database_session: MagicMock) -> None:
    """Новая запись добавляется в БД и кэш."""
    database_session.execute.return_value = _scalar_result(None)

    message = await emoji_descriptions_cache.save(
        name="heart", description="Сердце", guild_id=100
    )

    assert message == "Описание эмодзи 'heart' успешно добавлено!"
    database_session.add.assert_called_once()
    added = database_session.add.call_args.args[0]
    assert isinstance(added, GuildEmoji)
    assert added.name == "heart"
    assert added.description == "Сердце"
    assert added.guild_id == 100
    database_session.commit.assert_awaited_once()
    assert emoji_descriptions_cache.get(100) == {"heart": "Сердце"}


@pytest.mark.asyncio
async def test_remove_reports_success_when_deleted_from_database(
    database_session: MagicMock,
) -> None:
    """Удаление записи только из БД возвращает сообщение об успехе."""
    database_session.execute.return_value = MagicMock(rowcount=1)

    message = await emoji_descriptions_cache.remove(name="smile", guild_id=100)

    assert message == "Описание эмодзи 'smile' удалено."
    database_session.execute.assert_awaited_once()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_reports_success_when_deleted_from_cache(
    database_session: MagicMock,
) -> None:
    """Удаление записи только из кэша также считается успешным."""
    emoji_descriptions_cache._cache[100] = {"smile": "Улыбка"}
    database_session.execute.return_value = MagicMock(rowcount=0)

    message = await emoji_descriptions_cache.remove(name="smile", guild_id=100)

    assert message == "Описание эмодзи 'smile' удалено."
    assert emoji_descriptions_cache.get(100) == {}


@pytest.mark.asyncio
async def test_remove_reports_missing_when_row_and_cache_are_absent(
    database_session: MagicMock,
) -> None:
    """Если записи нет в БД и кэше, возвращается сообщение об отсутствии."""
    database_session.execute.return_value = MagicMock(rowcount=0)

    message = await emoji_descriptions_cache.remove(name="smile", guild_id=100)

    assert message == "Описание эмодзи 'smile' не найдено."


@pytest.mark.asyncio
async def test_load_all_wraps_database_error_and_preserves_cache(
    database_session: MagicMock,
) -> None:
    """Ошибка загрузки преобразуется декоратором и не очищает кэш."""
    emoji_descriptions_cache._cache[100] = {"smile": "Улыбка"}
    database_session.execute.side_effect = SQLAlchemyError("execute failed")

    with pytest.raises(RuntimeError) as error:
        await emoji_descriptions_cache.load_all()

    assert str(error.value) == "Ошибка базы данных при загрузке описаний эмодзи: execute failed"
    assert emoji_descriptions_cache.get(100) == {"smile": "Улыбка"}
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_save_wraps_commit_error_and_does_not_update_cache(
    database_session: MagicMock,
) -> None:
    """Ошибка commit преобразуется декоратором до обновления кэша."""
    database_session.execute.return_value = _scalar_result(None)
    database_session.commit.side_effect = SQLAlchemyError("commit failed")

    with pytest.raises(RuntimeError) as error:
        await emoji_descriptions_cache.save(
            name="smile", description="Улыбка", guild_id=100
        )

    assert str(error.value) == (
        "Ошибка базы данных при сохранении описания эмодзи: commit failed"
    )
    assert emoji_descriptions_cache.get(100) == {}
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_remove_wraps_database_error_and_preserves_cache(
    database_session: MagicMock,
) -> None:
    """Ошибка удаления преобразуется декоратором до изменения кэша."""
    emoji_descriptions_cache._cache[100] = {"smile": "Улыбка"}
    database_session.execute.side_effect = SQLAlchemyError("delete failed")

    with pytest.raises(RuntimeError) as error:
        await emoji_descriptions_cache.remove(name="smile", guild_id=100)

    assert str(error.value) == "Ошибка базы данных при удалении описания эмодзи: delete failed"
    assert emoji_descriptions_cache.get(100) == {"smile": "Улыбка"}
    database_session.commit.assert_not_awaited()
    assert database_session.context_exited is True