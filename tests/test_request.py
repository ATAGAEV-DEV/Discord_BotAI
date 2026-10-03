"""Изолированные тесты для операций базы данных из app/data/request.py."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.data import decorators, request
from app.data.models import ChannelMessage, UserDescription, UserMessageStats


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


def _rows_result(*rows: SimpleNamespace | ChannelMessage) -> MagicMock:
    """Создаёт результат SQLAlchemy со списком строк."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    return result


def _scalar_result(value: object) -> MagicMock:
    """Создаёт результат SQLAlchemy с одной найденной или отсутствующей записью."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_save_channel_message_creates_model_and_commits(
    database_session: MagicMock,
) -> None:
    """Сообщение канала сохраняется в виде ChannelMessage и фиксируется."""
    result = await request.save_channel_message(
        channel_id=100,
        message_id=200,
        author="Alice",
        content="Привет",
    )

    assert result is None
    database_session.add.assert_called_once()
    saved_message = database_session.add.call_args.args[0]
    assert isinstance(saved_message, ChannelMessage)
    assert saved_message.channel_id == 100
    assert saved_message.message_id == 200
    assert saved_message.author == "Alice"
    assert saved_message.content == "Привет"
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_channel_messages_returns_all_rows(database_session: MagicMock) -> None:
    """Возвращаются все сообщения, полученные из результата запроса."""
    messages = [
        ChannelMessage(channel_id=100, message_id=1, author="Alice", content="Первое"),
        ChannelMessage(channel_id=100, message_id=2, author="Bob", content="Второе"),
    ]
    database_session.execute.return_value = _rows_result(*messages)

    result = await request.get_channel_messages(100)

    assert result == messages
    database_session.execute.assert_awaited_once()
    database_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_channel_messages_executes_delete_and_commits(
    database_session: MagicMock,
) -> None:
    """Все сообщения канала удаляются с фиксацией транзакции."""
    result = await request.delete_channel_messages(100)

    assert result is None
    database_session.execute.assert_awaited_once()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_message_count_existing_user_without_rank_up(
    database_session: MagicMock,
) -> None:
    """Для существующего пользователя счётчик увеличивается без повышения ранга."""
    existing_stats = UserMessageStats(
        user_id=10,
        guild_id=100,
        name="Alice",
        message_count=1,
    )
    database_session.execute.return_value = _scalar_result(existing_stats)

    result = await request.update_message_count(10, "Alice", 100)

    assert result == {
        "rank_up": False,
        "old_rank": 1,
        "new_rank": 1,
        "message_count": 2,
    }
    assert database_session.execute.await_count == 2
    database_session.add.assert_not_called()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_message_count_existing_user_with_rank_up(
    database_session: MagicMock,
) -> None:
    """Переход с 49 на 50 сообщений отмечается как повышение ранга."""
    existing_stats = UserMessageStats(
        user_id=10,
        guild_id=100,
        name="Alice",
        message_count=49,
    )
    database_session.execute.return_value = _scalar_result(existing_stats)

    result = await request.update_message_count(10, "Alice", 100)

    assert result == {
        "rank_up": True,
        "old_rank": 1,
        "new_rank": 2,
        "message_count": 50,
    }
    assert database_session.execute.await_count == 2
    database_session.add.assert_not_called()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_message_count_creates_stats_for_new_user(
    database_session: MagicMock,
) -> None:
    """Для нового пользователя создаётся статистика с одним сообщением."""
    database_session.execute.return_value = _scalar_result(None)

    result = await request.update_message_count(10, "Alice", 100)

    assert result == {
        "rank_up": True,
        "old_rank": 0,
        "new_rank": 1,
        "message_count": 1,
    }
    database_session.execute.assert_awaited_once()
    database_session.add.assert_called_once()
    new_stats = database_session.add.call_args.args[0]
    assert isinstance(new_stats, UserMessageStats)
    assert new_stats.user_id == 10
    assert new_stats.guild_id == 100
    assert new_stats.name == "Alice"
    assert new_stats.message_count == 1
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_rank_returns_existing_message_count(database_session: MagicMock) -> None:
    """Для найденной записи возвращается её количество сообщений."""
    database_session.execute.return_value = _scalar_result(42)

    result = await request.get_rank(10, 100)

    assert result == 42
    database_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_rank_returns_zero_for_missing_user(database_session: MagicMock) -> None:
    """Для отсутствующей записи количество сообщений равно нулю."""
    database_session.execute.return_value = _scalar_result(None)

    result = await request.get_rank(10, 100)

    assert result == 0
    database_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_user_rank_returns_existing_rank(database_session: MagicMock) -> None:
    """Для найденного пользователя возвращается его место в рейтинге."""
    database_session.execute.return_value = _scalar_result(3)

    result = await request.get_user_rank(10, 100)

    assert result == 3
    database_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_user_rank_returns_zero_for_missing_user(database_session: MagicMock) -> None:
    """Для отсутствующего пользователя возвращается нулевой ранг."""
    database_session.execute.return_value = _scalar_result(None)

    result = await request.get_user_rank(10, 100)

    assert result == 0
    database_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_user_descriptions_returns_nick_description_mapping(
    database_session: MagicMock,
) -> None:
    """Описания преобразуются в словарь с никами в качестве ключей."""
    database_session.execute.return_value = _rows_result(
        SimpleNamespace(nick="Alice", description="Первое описание"),
        SimpleNamespace(nick="Bob", description="Второе описание"),
    )

    result = await request.get_user_descriptions(100)

    assert result == {
        "Alice": "Первое описание",
        "Bob": "Второе описание",
    }
    database_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_user_description_adds_new_entry(database_session: MagicMock) -> None:
    """Новое описание добавляется в базу данных."""
    database_session.execute.return_value = _scalar_result(None)

    result = await request.save_user_description("Alice", "Описание", 100)

    assert result == "Описание для 'Alice' успешно добавлено!"
    database_session.add.assert_called_once()
    new_description = database_session.add.call_args.args[0]
    assert isinstance(new_description, UserDescription)
    assert new_description.nick == "Alice"
    assert new_description.description == "Описание"
    assert new_description.guild_id == 100
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_user_description_updates_existing_entry(
    database_session: MagicMock,
) -> None:
    """Для существующего ника обновляется текст описания."""
    existing_description = UserDescription(
        nick="Alice",
        description="Старое описание",
        guild_id=100,
    )
    database_session.execute.return_value = _scalar_result(existing_description)

    result = await request.save_user_description("Alice", "Новое описание", 100)

    assert result == "Описание для 'Alice' успешно обновлено!"
    assert database_session.execute.await_count == 2
    lookup, write = [call.args[0] for call in database_session.execute.await_args_list]
    assert lookup.compare(
        select(UserDescription).where(
            UserDescription.nick == "Alice", UserDescription.guild_id == 100
        )
    )
    assert write.compare(
        update(UserDescription)
        .where(UserDescription.nick == "Alice", UserDescription.guild_id == 100)
        .values(description="Новое описание")
    )
    database_session.add.assert_not_called()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_user_description_returns_success_when_row_deleted(
    database_session: MagicMock,
) -> None:
    """Удаление существующей записи возвращает сообщение об успехе."""
    database_session.execute.return_value = MagicMock(rowcount=1)

    result = await request.delete_user_description("Alice", 100)

    assert result == "Описание для 'Alice' удалено."
    database_session.execute.assert_awaited_once()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_user_description_returns_not_found_when_row_missing(
    database_session: MagicMock,
) -> None:
    """При отсутствии записи возвращается соответствующее сообщение."""
    database_session.execute.return_value = MagicMock(rowcount=0)

    result = await request.delete_user_description("Alice", 100)

    assert result == "Описание для 'Alice' не найдено."
    database_session.execute.assert_awaited_once()
    database_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_request_wraps_sqlalchemy_error_and_closes_session(
    database_session: MagicMock,
) -> None:
    """Ошибка SQLAlchemy преобразуется декоратором и закрывает контекст сессии."""
    database_session.execute.side_effect = SQLAlchemyError("execute failed")

    with pytest.raises(RuntimeError) as error:
        await request.get_rank(10, 100)

    assert str(error.value) == (
        "Ошибка базы данных при получении статистики сообщений: execute failed"
    )
    assert database_session.context_entered is True
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_request_translates_operational_error(database_session: MagicMock) -> None:
    """Ошибка подключения к БД преобразуется в TimeoutError."""
    database_session.execute.side_effect = OperationalError(
        "select failed", {}, Exception("connection failed")
    )

    with pytest.raises(TimeoutError) as error:
        await request.get_rank(10, 100)

    assert str(error.value) == "Ошибка подключения/таймаут БД при получении статистики сообщений."
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_request_wraps_unexpected_error(database_session: MagicMock) -> None:
    """Непредвиденная ошибка преобразуется в RuntimeError."""
    database_session.execute.side_effect = ValueError("unexpected failure")

    with pytest.raises(RuntimeError) as error:
        await request.get_rank(10, 100)

    assert str(error.value) == (
        "Непредвиденная ошибка при получении статистики сообщений: unexpected failure"
    )
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_save_channel_message_wraps_commit_error(database_session: MagicMock) -> None:
    """Ошибка commit при сохранении сообщения преобразуется декоратором."""
    database_session.commit.side_effect = SQLAlchemyError("commit failed")

    with pytest.raises(RuntimeError) as error:
        await request.save_channel_message(100, 200, "Alice", "Привет")

    assert str(error.value) == "Ошибка базы данных при сохранении сообщения канала: commit failed"
    assert database_session.add.call_count == 1
    assert database_session.context_exited is True