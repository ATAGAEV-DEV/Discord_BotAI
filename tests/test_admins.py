"""Тесты кэша и операций с администраторами бота."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.data import admins, decorators
from app.data.models import BotAdmin


@pytest.fixture(autouse=True)
def empty_admin_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует глобальный кэш администраторов между тестами."""
    monkeypatch.setattr(admins, "_cache", {})


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
    """Создаёт результат SQLAlchemy со строками администраторов."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    return result


def _scalar_result(value: object) -> MagicMock:
    """Создаёт результат SQLAlchemy с одной найденной или отсутствующей записью."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_load_all_replaces_cache_and_groups_rows(
    database_session: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Загрузка очищает старые данные и группирует администраторов по серверам."""
    admins._cache[999] = {1: "Stale"}
    database_session.execute.return_value = _rows_result(
        SimpleNamespace(guild_id=100, user_id=10, username="Alice"),
        SimpleNamespace(guild_id=100, user_id=20, username="Bob"),
        SimpleNamespace(guild_id=200, user_id=30, username="Charlie"),
    )

    result = await admins.load_all()

    assert result is None
    database_session.execute.assert_awaited_once()
    assert admins.get_all(100) == {10: "Alice", 20: "Bob"}
    assert admins.get_all(200) == {30: "Charlie"}
    assert admins.get_all(999) == {}
    assert "Кэш администраторов загружен: 3 записей" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_load_all_with_empty_result_clears_cache(
    database_session: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Пустой результат загрузки удаляет администраторов из старого кэша."""
    admins._cache[100] = {10: "Alice"}
    database_session.execute.return_value = _rows_result()

    await admins.load_all()

    assert admins._cache == {}
    assert "Кэш администраторов загружен: 0 записей" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_load_all_uses_last_username_for_duplicate_user(
    database_session: MagicMock,
) -> None:
    """Для повторной пары server/user в кэше остаётся последнее имя."""
    database_session.execute.return_value = _rows_result(
        SimpleNamespace(guild_id=100, user_id=10, username="OldName"),
        SimpleNamespace(guild_id=100, user_id=10, username="NewName"),
    )

    await admins.load_all()

    assert admins.get_all(100) == {10: "NewName"}


@pytest.mark.parametrize(
    ("guild_id", "user_id", "expected"),
    [
        (100, 10, True),
        (100, 20, False),
        (999, 10, False),
    ],
)
def test_is_admin_checks_user_and_guild(
    guild_id: int, user_id: int, expected: bool
) -> None:
    """Проверка учитывает и сервер, и идентификатор пользователя."""
    admins._cache[100] = {10: "Alice"}

    assert admins.is_admin(guild_id, user_id) is expected


def test_get_all_returns_empty_for_unknown_guild() -> None:
    """Для неизвестного сервера возвращается пустой словарь."""
    admins._cache[100] = {10: "Alice"}

    assert admins.get_all(999) == {}


def test_get_all_returns_copy_of_guild_cache() -> None:
    """Изменение результата get_all не изменяет внутренний кэш."""
    admins._cache[100] = {10: "Alice"}

    result = admins.get_all(100)
    result[20] = "Bob"

    assert result == {10: "Alice", 20: "Bob"}
    assert admins.get_all(100) == {10: "Alice"}
    assert admins.is_admin(100, 20) is False


@pytest.mark.asyncio
async def test_add_creates_new_admin_and_updates_cache(database_session: MagicMock) -> None:
    """Новый администратор добавляется в БД и кэш."""
    database_session.execute.return_value = _scalar_result(None)

    message = await admins.add(guild_id=100, user_id=10, username="Alice")

    database_session.execute.assert_awaited_once()
    database_session.add.assert_called_once()
    added_admin = database_session.add.call_args.args[0]
    assert isinstance(added_admin, BotAdmin)
    assert added_admin.guild_id == 100
    assert added_admin.user_id == 10
    assert added_admin.username == "Alice"
    database_session.commit.assert_awaited_once()
    assert admins.get_all(100) == {10: "Alice"}
    assert message == "✅ администратор добавлен: **Alice** (10)"


@pytest.mark.asyncio
async def test_add_updates_existing_admin_name(database_session: MagicMock) -> None:
    """Для существующего администратора выполняется обновление имени."""
    existing_admin = BotAdmin(guild_id=100, user_id=10, username="OldName")
    database_session.execute.side_effect = [
        _scalar_result(existing_admin),
        MagicMock(),
    ]

    message = await admins.add(guild_id=100, user_id=10, username="NewName")

    assert database_session.execute.await_count == 2
    database_session.add.assert_not_called()
    database_session.commit.assert_awaited_once()
    assert admins.get_all(100) == {10: "NewName"}
    assert message == "✅ имя администратора обновлено: **NewName** (10)"


@pytest.mark.asyncio
async def test_remove_returns_success_when_database_row_was_deleted(
    database_session: MagicMock,
) -> None:
    """Удаление записи из базы считается успешным даже без записи в кэше."""
    result = MagicMock(rowcount=1)
    database_session.execute.return_value = result

    message = await admins.remove(guild_id=100, user_id=10)

    database_session.execute.assert_awaited_once()
    database_session.commit.assert_awaited_once()
    assert message == "✅ Администратор **10** удалён."


@pytest.mark.asyncio
async def test_remove_returns_success_when_only_cache_row_was_deleted(
    database_session: MagicMock,
) -> None:
    """Удаление записи только из кэша также считается успешным."""
    admins._cache[100] = {10: "Alice"}
    database_session.execute.return_value = MagicMock(rowcount=0)

    message = await admins.remove(guild_id=100, user_id=10)

    assert message == "✅ Администратор **10** удалён."
    assert admins.get_all(100) == {}


@pytest.mark.asyncio
async def test_remove_reports_missing_admin(database_session: MagicMock) -> None:
    """Если записи нет ни в базе, ни в кэше, возвращается информационное сообщение."""
    database_session.execute.return_value = MagicMock(rowcount=0)

    message = await admins.remove(guild_id=100, user_id=10)

    database_session.commit.assert_awaited_once()
    assert message == "ℹ️ Администратор **10** не найден."


@pytest.mark.asyncio
async def test_admins_are_isolated_between_guilds(database_session: MagicMock) -> None:
    """Удаление администратора одного сервера не затрагивает другой сервер."""
    database_session.execute.side_effect = [
        _scalar_result(None),
        _scalar_result(None),
        MagicMock(rowcount=1),
    ]

    await admins.add(guild_id=100, user_id=10, username="GuildOne")
    await admins.add(guild_id=200, user_id=10, username="GuildTwo")
    await admins.remove(guild_id=100, user_id=10)

    assert admins.is_admin(100, 10) is False
    assert admins.is_admin(200, 10) is True
    assert admins.get_all(200) == {10: "GuildTwo"}


@pytest.mark.asyncio
async def test_load_all_wraps_sqlalchemy_error_and_closes_session(
    database_session: MagicMock,
) -> None:
    """Ошибка SQLAlchemy преобразуется декоратором и не оставляет сессию открытой."""
    database_session.execute.side_effect = SQLAlchemyError("execute failed")

    with pytest.raises(RuntimeError) as error:
        await admins.load_all()

    assert str(error.value) == (
        "Ошибка базы данных при загрузке администраторов бота: execute failed"
    )
    assert database_session.context_entered is True
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_add_wraps_commit_error_and_does_not_update_cache(
    database_session: MagicMock,
) -> None:
    """Ошибка commit возвращается вызывающему коду, а кэш не обновляется."""
    database_session.execute.return_value = _scalar_result(None)
    database_session.commit.side_effect = SQLAlchemyError("commit failed")

    with pytest.raises(RuntimeError) as error:
        await admins.add(guild_id=100, user_id=10, username="Alice")

    assert str(error.value) == (
        "Ошибка базы данных при добавлении администратора бота: commit failed"
    )
    assert admins.get_all(100) == {}
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_load_all_translates_timeout_error(database_session: MagicMock) -> None:
    """Встроенный TimeoutError получает сообщение с названием операции."""
    database_session.execute.side_effect = TimeoutError()

    with pytest.raises(TimeoutError) as error:
        await admins.load_all()

    assert str(error.value) == "Таймаут при загрузке администраторов бота."
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_load_all_translates_operational_error(database_session: MagicMock) -> None:
    """Ошибка подключения SQLAlchemy преобразуется в TimeoutError."""
    database_session.execute.side_effect = OperationalError(
        "select failed", {}, Exception("connection failed")
    )

    with pytest.raises(TimeoutError) as error:
        await admins.load_all()

    assert str(error.value) == "Ошибка подключения/таймаут БД при загрузке администраторов бота."
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_load_all_wraps_unexpected_error(database_session: MagicMock) -> None:
    """Непредвиденная ошибка получает общий RuntimeError от декоратора."""
    database_session.execute.side_effect = ValueError("unexpected failure")

    with pytest.raises(RuntimeError) as error:
        await admins.load_all()

    assert str(error.value) == (
        "Непредвиденная ошибка при загрузке администраторов бота: unexpected failure"
    )
    assert database_session.context_exited is True