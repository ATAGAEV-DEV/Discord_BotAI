"""CRUD, PostgreSQL rank(), кэши и декораторы поверх настоящих двух БД."""

from types import ModuleType

import pytest
from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError

from app.data import (
    admins,
    decorators,
    emoji_descriptions_cache,
    models,
    request,
    user_descriptions_cache,
)
from tests.integration.conftest import Databases


async def test_channel_messages_crud_is_isolated_by_channel(databases: Databases) -> None:
    """Сообщения записываются в обе БД; удаление не затрагивает другой канал."""
    await request.save_channel_message(100, 1001, "alice", "Привет 👋")
    await request.save_channel_message(100, 1002, "bob", "Ответ")
    await request.save_channel_message(200, 2001, "alice", "Другой канал")
    assert {row.message_id for row in await request.get_channel_messages(100)} == {1001, 1002}
    for side in ("local", "remote"):
        assert {
            (r.channel_id, r.message_id, r.content)
            for r in await databases.rows(side, models.ChannelMessage)
        } == {
            (100, 1001, "Привет 👋"),
            (100, 1002, "Ответ"),
            (200, 2001, "Другой канал"),
        }
    await request.delete_channel_messages(100)
    assert await request.get_channel_messages(100) == []
    for side in ("local", "remote"):
        row, = await databases.rows(side, models.ChannelMessage)
        assert row.message_id == 2001


async def test_message_counters_are_committed_per_guild(databases: Databases) -> None:
    """Первый INSERT и последующие UPDATE обновляют только нужную пару user/guild."""
    assert await request.get_rank(1, 100) == 0
    first = await request.update_message_count(1, "alice", 100)
    second = await request.update_message_count(1, "alice", 100)
    await request.update_message_count(1, "alice", 200)
    assert first["message_count"] == 1
    assert second["message_count"] == 2
    assert await request.get_rank(1, 100) == 2
    assert await request.get_rank(1, 200) == 1
    for side in ("local", "remote"):
        assert {
            (row.user_id, row.guild_id): row.message_count
            for row in await databases.rows(side, models.UserMessageStats)
        } == {(1, 100): 2, (1, 200): 1}


async def test_postgres_rank_ties_gaps_and_guild_partition(databases: Databases) -> None:
    """Одинаковые счётчики получают rank 1,1,3, а не row_number или dense_rank."""
    for side in ("local", "remote"):
        await databases.seed(
            side,
            *[
                models.UserMessageStats(
                    user_id=user, guild_id=guild, name=str(user), message_count=count
                )
                for user, guild, count in [(1, 100, 10), (2, 100, 10), (3, 100, 2), (3, 200, 999)]
            ],
        )
    assert await request.get_user_rank(1, 100) == 1
    assert await request.get_user_rank(2, 100) == 1
    assert await request.get_user_rank(3, 100) == 3
    assert await request.get_user_rank(3, 200) == 1
    assert await request.get_user_rank(99, 100) == 0
    assert await request.get_user_rank(1, 999) == 0


async def test_request_descriptions_update_and_delete_both_databases(databases: Databases) -> None:
    """Request CRUD и явное обновление описания согласованы в обеих БД."""
    assert "добавлено" in await request.save_user_description("alice", "old", 100)
    await request.save_user_description("alice", "other guild", 200)
    async with models.async_session() as session:
        await session.execute(
            update(models.UserDescription)
            .where(
                models.UserDescription.nick == "alice",
                models.UserDescription.guild_id == 100,
            )
            .values(description="new")
        )
        await session.commit()
    assert await request.get_user_descriptions(100) == {"alice": "new"}
    for side in ("local", "remote"):
        assert {
            r.guild_id: r.description for r in await databases.rows(side, models.UserDescription)
        } == {100: "new", 200: "other guild"}
    assert "удалено" in await request.delete_user_description("alice", 100)
    assert "не найдено" in await request.delete_user_description("alice", 100)
    for side in ("local", "remote"):
        row, = await databases.rows(side, models.UserDescription)
        assert row.guild_id == 200


@pytest.mark.parametrize(
    "cache, model, key",
    [
        (user_descriptions_cache, models.UserDescription, "nick"),
        (emoji_descriptions_cache, models.GuildEmoji, "name"),
    ],
)
async def test_description_cache_crud_reload_and_copies(
    databases: Databases, cache: ModuleType, model: type[models.Base], key: str
) -> None:
    """Кэш поддерживает CRUD, reload и копии без смешения guild."""
    await cache.save("same", "old", 100)
    await cache.save("same", "other guild", 200)
    assert cache.get(100) == {"same": "old"}
    assert cache.get(200) == {"same": "other guild"}
    copy = cache.get(100)
    copy["same"] = "mutated"
    assert cache.get(100) == {"same": "old"}
    for side in ("local", "remote"):
        assert {
            (r.guild_id, getattr(r, key)): r.description for r in await databases.rows(side, model)
        } == {(100, "same"): "old", (200, "same"): "other guild"}
    async with models.async_session() as session:
        await session.execute(
            update(model)
            .where(getattr(model, key) == "same", model.guild_id == 100)
            .values(description="new")
        )
        await session.commit()
    cache._cache[999] = {"stale": "discard"}
    await cache.load_all()
    assert cache.get(999) == {}
    assert cache.get(100) == {"same": "new"}
    assert "удалено" in await cache.remove("same", 100)
    assert "не найдено" in await cache.remove("same", 100)
    assert cache.get(100) == {}
    for side in ("local", "remote"):
        row, = await databases.rows(side, model)
        assert row.guild_id == 200


async def test_admin_crud_reload_and_server_isolation(databases: Databases) -> None:
    """Администраторы и их имена согласованы с двумя таблицами и локальным кэшем."""
    await admins.add(100, 1, "old")
    await admins.add(200, 1, "other guild")
    await admins.add(100, 1, "new")
    assert admins.is_admin(100, 1)
    assert not admins.is_admin(300, 1)
    copy = admins.get_all(100)
    copy.clear()
    assert admins.get_all(100) == {1: "new"}
    for side in ("local", "remote"):
        assert {
            (r.guild_id, r.user_id): r.username for r in await databases.rows(side, models.BotAdmin)
        } == {(100, 1): "new", (200, 1): "other guild"}
    admins._cache[999] = {99: "stale"}
    await admins.load_all()
    assert admins.get_all(999) == {}
    assert admins.get_all(100) == {1: "new"}
    assert "удалён" in await admins.remove(100, 1)
    assert "не найден" in await admins.remove(100, 1)
    assert not admins.is_admin(100, 1)
    assert admins.is_admin(200, 1)
    for side in ("local", "remote"):
        row, = await databases.rows(side, models.BotAdmin)
        assert row.guild_id == 200


async def test_decorator_wraps_integrity_error_and_releases_connections(
    databases: Databases,
) -> None:
    """Реальный NOT NULL приводит к RuntimeError; открытые сессии не остаются в pool."""
    with pytest.raises(RuntimeError, match="Ошибка базы данных") as error:
        await request.save_channel_message(100, 1, "alice", None)
    assert isinstance(error.value.__cause__, IntegrityError)
    assert all(engine.pool.checkedout() == 0 for engine in databases.engines)
    for side in ("local", "remote"):
        assert await databases.rows(side, models.ChannelMessage) == []
    await request.save_channel_message(100, 2, "alice", "recovered")
    for side in ("local", "remote"):
        row, = await databases.rows(side, models.ChannelMessage)
        assert row.message_id == 2


async def test_decorator_timeout_closes_cancelled_database_session(
    databases: Databases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Клиент отменяет один короткий SELECT; серверы и чужие подключения не трогаем."""
    monkeypatch.setattr(decorators, "DB_TIMEOUT", 0.05)

    @decorators.db_operation("тестовом ожидании")
    async def slow_query(session: models.DualSessionProxy) -> None:
        await session.execute(text("SELECT pg_sleep(0.2)"))

    with pytest.raises(TimeoutError, match="Таймаут при тестовом ожидании"):
        await slow_query()
    assert all(engine.pool.checkedout() == 0 for engine in databases.engines)
    for maker in (databases.local, databases.remote):
        async with maker() as session:
            assert await session.scalar(text("SELECT 1")) == 1