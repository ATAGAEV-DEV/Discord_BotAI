"""Изолированные тесты кэша описаний пользователей."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import delete as sa_delete
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from app.data import decorators, user_descriptions_cache
from app.data.models import UserDescription


@pytest.fixture(autouse=True)
def empty_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолирует кэш и флаг загрузки, восстанавливая их после каждого теста."""
    monkeypatch.setattr(user_descriptions_cache, "_cache", {})
    monkeypatch.setattr(user_descriptions_cache, "_loaded", False)


@pytest.fixture
def database_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Подменяет сессию декоратора без подключения к реальной БД."""
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



def _rows_result(*rows: UserDescription) -> MagicMock:
    """Создаёт результат SQLAlchemy со списком описаний пользователей."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    return result


def _scalar_result(value: UserDescription | None) -> MagicMock:
    """Создаёт результат поиска существующего или отсутствующего описания."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_load_all_replaces_cache_and_groups_rows(
    database_session: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Загрузка заменяет старый кэш и группирует описания по серверам."""
    user_descriptions_cache._cache[999] = {"Old": "Устаревшее описание"}
    database_session.execute.return_value = _rows_result(
        UserDescription(guild_id=100, nick="Alice", description="Алиса"),
        UserDescription(guild_id=100, nick="Bob", description="Боб"),
        UserDescription(guild_id=200, nick="Alice", description="Другая Алиса"),
    )

    result = await user_descriptions_cache.load_all()

    assert result is None
    database_session.execute.assert_awaited_once()
    query = database_session.execute.call_args.args[0]
    assert query.compare(select(UserDescription))
    database_session.commit.assert_not_awaited()
    assert user_descriptions_cache._cache == {
        100: {"Alice": "Алиса", "Bob": "Боб"},
        200: {"Alice": "Другая Алиса"},
    }
    assert user_descriptions_cache._loaded is True
    assert "Кэш описаний загружен: 3 записей" in capsys.readouterr().out
    assert database_session.context_entered is True
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_load_all_with_empty_result_clears_cache(
    database_session: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Пустой результат очищает кэш и отмечает завершение загрузки."""
    user_descriptions_cache._cache[100] = {"Alice": "Алиса"}
    database_session.execute.return_value = _rows_result()

    await user_descriptions_cache.load_all()

    assert user_descriptions_cache._cache == {}
    assert user_descriptions_cache._loaded is True
    assert "Кэш описаний загружен: 0 записей" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_load_all_uses_last_description_for_duplicate_nick(
    database_session: MagicMock,
) -> None:
    """Последняя запись для пары сервер/ник заменяет предыдущую."""
    database_session.execute.return_value = _rows_result(
        UserDescription(guild_id=100, nick="Alice", description="Старое описание"),
        UserDescription(guild_id=100, nick="Alice", description="Новое описание"),
    )

    await user_descriptions_cache.load_all()

    assert user_descriptions_cache.get(100) == {"Alice": "Новое описание"}


@pytest.mark.parametrize("guild_id", [0, 999])
def test_get_returns_empty_dict_for_unknown_guild(guild_id: int) -> None:
    """Чтение отсутствующего сервера возвращает пустой словарь без изменения кэша."""
    assert user_descriptions_cache.get(guild_id) == {}
    assert user_descriptions_cache._cache == {}


def test_get_returns_only_requested_guild() -> None:
    """Описания другого сервера не попадают в результат чтения."""
    user_descriptions_cache._cache.update(
        {100: {"Alice": "Алиса"}, 200: {"Alice": "Другая Алиса", "Bob": "Боб"}}
    )

    assert user_descriptions_cache.get(100) == {"Alice": "Алиса"}
    assert user_descriptions_cache.get(200) == {"Alice": "Другая Алиса", "Bob": "Боб"}


def test_get_returns_copy_without_exposing_cache() -> None:
    """Изменение возвращённого словаря не изменяет внутренний кэш."""
    user_descriptions_cache._cache[100] = {"Alice": "Алиса"}

    descriptions = user_descriptions_cache.get(100)
    descriptions["Alice"] = "Изменено"
    descriptions["Bob"] = "Боб"

    assert user_descriptions_cache.get(100) == {"Alice": "Алиса"}


def test_get_all_returns_empty_dict_for_empty_cache() -> None:
    """Пустой кэш возвращается в виде пустого объединённого словаря."""
    assert user_descriptions_cache.get_all() == {}


def test_get_all_merges_guilds_and_replaces_conflicting_nicks() -> None:
    """Описания объединяются, а повторный ник получает последнее описание."""
    user_descriptions_cache._cache.update(
        {
            100: {"Alice": "Первая Алиса", "Bob": "Боб"},
            200: {"Alice": "Другая Алиса", "Charlie": "Чарли"},
        }
    )

    assert user_descriptions_cache.get_all() == {
        "Alice": "Другая Алиса", "Bob": "Боб", "Charlie": "Чарли"
    }


def test_get_all_returns_copy_without_exposing_cache() -> None:
    """Изменение объединённого словаря не затрагивает кэш серверов."""
    user_descriptions_cache._cache[100] = {"Alice": "Алиса"}

    descriptions = user_descriptions_cache.get_all()
    descriptions.clear()

    assert user_descriptions_cache.get(100) == {"Alice": "Алиса"}


@pytest.mark.asyncio
async def test_save_updates_existing_row_and_cache(database_session: MagicMock) -> None:
    """Существующая запись обновляется только для выбранного сервера."""
    existing = UserDescription(guild_id=100, nick="Alice", description="Старое описание")
    database_session.execute.return_value = _scalar_result(existing)
    user_descriptions_cache._cache.update(
        {
            100: {"Alice": "Старое описание", "Bob": "Боб"},
            200: {"Alice": "Другая Алиса"},
        }
    )

    message = await user_descriptions_cache.save(
        nick="Alice", description="Новое описание", guild_id=100
    )

    assert message == "Описание для 'Alice' успешно обновлено!"
    assert database_session.execute.await_count == 2
    query, write = [call.args[0] for call in database_session.execute.await_args_list]
    assert query.compare(
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
    assert user_descriptions_cache.get(100) == {"Alice": "Новое описание", "Bob": "Боб"}
    assert user_descriptions_cache.get(200) == {"Alice": "Другая Алиса"}
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_save_adds_new_row_and_guild_cache(database_session: MagicMock) -> None:
    """Новая запись добавляется в БД и создаёт отсутствующий кэш сервера."""
    database_session.execute.return_value = _scalar_result(None)
    user_descriptions_cache._cache[200] = {"Alice": "Другая Алиса"}

    message = await user_descriptions_cache.save(
        nick="Alice", description="Алиса", guild_id=100
    )

    assert message == "Описание для 'Alice' успешно добавлено!"
    database_session.execute.assert_awaited_once()
    database_session.add.assert_called_once()
    added = database_session.add.call_args.args[0]
    assert isinstance(added, UserDescription)
    assert added.nick == "Alice"
    assert added.description == "Алиса"
    assert added.guild_id == 100
    database_session.commit.assert_awaited_once()
    assert user_descriptions_cache.get(100) == {"Alice": "Алиса"}
    assert user_descriptions_cache.get(200) == {"Alice": "Другая Алиса"}
    assert database_session.context_exited is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rowcount", "cached_description", "guild_exists"),
    [
        pytest.param(1, "Алиса", True, id="database-and-cache"),
        pytest.param(1, None, False, id="database-only"),
        pytest.param(0, "Алиса", True, id="cache-only"),
        pytest.param(0, "", True, id="empty-description-in-cache"),
        pytest.param(0, None, True, id="missing-nick"),
        pytest.param(0, None, False, id="missing-guild"),
    ],
)
async def test_remove_reports_result_and_preserves_other_descriptions(
    database_session: MagicMock,
    rowcount: int,
    cached_description: str | None,
    guild_exists: bool,
) -> None:
    """Удаление учитывает БД и кэш, не затрагивая другие ники и серверы."""
    user_descriptions_cache._cache[200] = {"Alice": "Другая Алиса"}
    if guild_exists:
        user_descriptions_cache._cache[100] = {"Bob": "Боб"}
        if cached_description is not None:
            user_descriptions_cache._cache[100]["Alice"] = cached_description
    database_session.execute.return_value = MagicMock(rowcount=rowcount)

    message = await user_descriptions_cache.remove(nick="Alice", guild_id=100)

    expected = "удалено" if rowcount > 0 or cached_description is not None else "не найдено"
    assert message == f"Описание для 'Alice' {expected}."
    database_session.execute.assert_awaited_once()
    query = database_session.execute.call_args.args[0]
    assert query.compare(
        sa_delete(UserDescription).where(
            UserDescription.nick == "Alice", UserDescription.guild_id == 100
        )
    )
    database_session.commit.assert_awaited_once()
    assert user_descriptions_cache.get(100) == ({"Bob": "Боб"} if guild_exists else {})
    assert user_descriptions_cache.get(200) == {"Alice": "Другая Алиса"}
    assert (100 in user_descriptions_cache._cache) is guild_exists
    assert database_session.context_exited is True


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded", [False, True])
async def test_load_all_wraps_database_error_and_preserves_state(
    database_session: MagicMock, loaded: bool
) -> None:
    """Ошибка загрузки сохраняет старые описания и значение флага загрузки."""
    user_descriptions_cache._cache[100] = {"Alice": "Алиса"}
    user_descriptions_cache._loaded = loaded
    database_session.execute.side_effect = SQLAlchemyError("execute failed")

    with pytest.raises(RuntimeError) as error:
        await user_descriptions_cache.load_all()

    assert str(error.value) == (
        "Ошибка базы данных при загрузке описаний пользователей: execute failed"
    )
    assert isinstance(error.value.__cause__, SQLAlchemyError)
    assert user_descriptions_cache.get(100) == {"Alice": "Алиса"}
    assert user_descriptions_cache._loaded is loaded
    database_session.commit.assert_not_awaited()
    assert database_session.context_exited is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["execute", "commit"])
@pytest.mark.parametrize("has_existing", [False, True], ids=["new", "existing"])
async def test_save_wraps_database_error_without_updating_cache(
    database_session: MagicMock, stage: str, has_existing: bool
) -> None:
    """Ошибка поиска или фиксации не меняет кэш при добавлении и обновлении."""
    existing = (
        UserDescription(guild_id=100, nick="Alice", description="Старое описание")
        if has_existing else None
    )
    database_session.execute.return_value = _scalar_result(existing)
    getattr(database_session, stage).side_effect = SQLAlchemyError(f"{stage} failed")
    user_descriptions_cache._cache.update(
        {100: {"Alice": "Старое описание"}, 200: {"Alice": "Другая Алиса"}}
    )

    with pytest.raises(RuntimeError) as error:
        await user_descriptions_cache.save(
            nick="Alice", description="Новое описание", guild_id=100
        )

    assert str(error.value) == (
        f"Ошибка базы данных при сохранении описания пользователя: {stage} failed"
    )
    assert isinstance(error.value.__cause__, SQLAlchemyError)
    assert user_descriptions_cache._cache == {
        100: {"Alice": "Старое описание"}, 200: {"Alice": "Другая Алиса"}
    }
    if stage == "execute":
        database_session.add.assert_not_called()
        database_session.commit.assert_not_awaited()
    else:
        database_session.commit.assert_awaited_once()
    assert database_session.context_exited is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["execute", "commit"])
async def test_remove_wraps_database_error_without_updating_cache(
    database_session: MagicMock, stage: str
) -> None:
    """Ошибка удаления или фиксации не удаляет описание из кэша."""
    user_descriptions_cache._cache.update(
        {100: {"Alice": "Алиса"}, 200: {"Alice": "Другая Алиса"}}
    )
    database_session.execute.return_value = MagicMock(rowcount=1)
    getattr(database_session, stage).side_effect = SQLAlchemyError(f"{stage} failed")

    with pytest.raises(RuntimeError) as error:
        await user_descriptions_cache.remove(nick="Alice", guild_id=100)

    assert str(error.value) == (
        f"Ошибка базы данных при удалении описания пользователя: {stage} failed"
    )
    assert isinstance(error.value.__cause__, SQLAlchemyError)
    assert user_descriptions_cache._cache == {
        100: {"Alice": "Алиса"}, 200: {"Alice": "Другая Алиса"}
    }
    if stage == "execute":
        database_session.commit.assert_not_awaited()
    else:
        database_session.commit.assert_awaited_once()
    assert database_session.context_exited is True
