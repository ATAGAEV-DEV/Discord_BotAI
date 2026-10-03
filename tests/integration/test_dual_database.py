"""Реальные PostgreSQL-типы, ограничения и семантика двухбазового прокси."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.exc import IntegrityError, OperationalError

from app.data import admins, emoji_descriptions_cache, models, user_descriptions_cache
from tests.integration.conftest import Databases


async def test_schema_is_empty_and_connections_are_isolated(databases: Databases) -> None:
    """Каждый тест начинает с пустых таблиц, кэшей и своей схемы в каждой БД."""
    for maker in (databases.local, databases.remote):
        async with maker() as session:
            assert await session.scalar(text("SELECT current_database()")) == "test_atagaev"
            assert await session.scalar(text("SELECT current_schema()")) == databases.schema
            assert await session.scalar(text("SHOW statement_timeout")) == "5s"
            for table in models.Base.metadata.sorted_tables:
                assert await session.scalar(select(func.count()).select_from(table)) == 0
    assert not admins._cache
    assert not emoji_descriptions_cache._cache
    assert not user_descriptions_cache._cache
    assert not user_descriptions_cache._loaded


async def test_bigint_jsonb_and_server_defaults_round_trip(databases: Databases) -> None:
    """Snowflake, вложенный JSONB, Unicode и серверные даты сохраняются в обе БД."""
    snowflake = 9223372036854775000
    context = {"dialog": [{"text": "Привет 👋", "ok": True}], "empty": None}
    async with models.async_session() as session:
        session.add(models.User(user_id=snowflake, name="Тест", context=context))
        session.add(models.UserMessageStats(user_id=snowflake, guild_id=100, name="Тест"))
        await session.commit()
    for side in ("local", "remote"):
        user, = await databases.rows(side, models.User)
        stats, = await databases.rows(side, models.UserMessageStats)
        assert user.user_id == snowflake
        assert user.context == context
        assert isinstance(user.datetime_insert, datetime)
        assert stats.message_count == 0
        assert isinstance(stats.last_updated, datetime)
        async with getattr(databases, side)() as session:
            column_type = await session.scalar(text("SELECT pg_typeof(context)::text FROM users"))
            assert column_type == "jsonb"


@pytest.mark.parametrize("side", ["local", "remote"])
async def test_unique_not_null_and_composite_constraints(databases: Databases, side: str) -> None:
    """Ограничения исполняет PostgreSQL; одинаковый ключ допустим в другом guild."""
    cases = [
        (models.UserDescription, {"nick": "alice", "description": "old"}),
        (models.GuildEmoji, {"name": "smile", "description": "old"}),
        (models.BotAdmin, {"user_id": 10, "username": "alice"}),
        (models.UserMessageStats, {"user_id": 10, "name": "alice"}),
        (
            models.YouTubeVideo,
            {"video_id": "v1", "channel_id": "UC1", "title": "Video",
             "published_at": datetime(2026, 1, 1)},
        ),
    ]
    for model, values in cases:
        await databases.seed(side, model(guild_id=100, **values), model(guild_id=200, **values))
        async with getattr(databases, side)() as session:
            session.add(model(guild_id=100, **values))
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
            assert await session.scalar(select(func.count()).select_from(model)) == 2
    async with getattr(databases, side)() as session:
        session.add(models.ChannelMessage(channel_id=1, message_id=2, author="a", content=None))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
        assert await session.scalar(select(func.count()).select_from(models.ChannelMessage)) == 0


async def test_reads_local_even_when_remote_data_differs(databases: Databases) -> None:
    """Remote не используется ни как основной источник, ни для слияния результатов."""
    for side in ("local", "remote"):
        await databases.seed(side, models.UserDescription(nick=side, description=side, guild_id=100))
    async with models.async_session() as session:
        rows = (await session.execute(select(models.UserDescription))).scalars().all()
        assert [row.nick for row in rows] == ["local"]


async def test_explicit_dml_is_committed_to_both_databases(databases: Databases) -> None:
    """INSERT/UPDATE/DELETE через execute видны новым независимым сессиям."""
    async with models.async_session() as session:
        await session.execute(insert(models.UserDescription).values(
            nick="alice", description="old", guild_id=100
        ))
        await session.commit()
        await session.execute(update(models.UserDescription).where(
            models.UserDescription.nick == "alice"
        ).values(description="new"))
        await session.commit()
    for side in ("local", "remote"):
        row, = await databases.rows(side, models.UserDescription)
        assert row.description == "new"
    async with models.async_session() as session:
        await session.execute(delete(models.UserDescription).where(
            models.UserDescription.nick == "alice"
        ))
        await session.commit()
    for side in ("local", "remote"):
        assert await databases.rows(side, models.UserDescription) == []


@pytest.mark.parametrize("explicit_rollback", [True, False])
async def test_uncommitted_writes_are_rolled_back(
    databases: Databases, explicit_rollback: bool
) -> None:
    """Явный rollback и выход из контекста отменяют уже выполненный DML в обеих БД."""
    async with models.async_session() as session:
        await session.execute(insert(models.UserDescription).values(
            nick="alice", description="uncommitted", guild_id=100
        ))
        if explicit_rollback:
            await session.rollback()
    for side in ("local", "remote"):
        assert await databases.rows(side, models.UserDescription) == []
    assert all(engine.pool.checkedout() == 0 for engine in databases.engines)


async def test_explicit_update_uses_logical_identity_with_different_ids(
    databases: Databases,
) -> None:
    """Явный UPDATE дублируется по бизнес-ключу, а не по serial ID локальной БД."""
    for side, row_id in (("local", 1), ("remote", 42)):
        await databases.seed(side, models.UserDescription(
            id=row_id, nick="alice", description="old", guild_id=100
        ))
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
    for side, expected_id in (("local", 1), ("remote", 42)):
        row, = await databases.rows(side, models.UserDescription)
        assert row.id == expected_id
        assert row.description == "new"


async def test_remote_commit_failure_keeps_local_commit_and_recovers(
    databases: Databases, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Фиксирует существующий best-effort контракт: общей атомарности нет."""
    async with models.async_session() as session:
        session.add(models.UserDescription(nick="first", description="local-only", guild_id=100))
        with monkeypatch.context() as patch:
            patch.setattr(
                session.remote,
                "commit",
                AsyncMock(side_effect=OperationalError(
                    "COMMIT", {}, RuntimeError("injected remote failure")
                )),
            )
            await session.commit()
        assert not session.remote.in_transaction()
        session.add(models.UserDescription(nick="second", description="both", guild_id=100))
        await session.commit()
    assert {row.nick for row in await databases.rows("local", models.UserDescription)} == {
        "first", "second"
    }
    assert [row.nick for row in await databases.rows("remote", models.UserDescription)] == ["second"]
    assert "Ошибка коммита в удаленной БД" in capsys.readouterr().out


async def test_remote_dml_failure_rolls_back_failed_transaction(
    databases: Databases, capsys: pytest.CaptureFixture[str]
) -> None:
    """Remote DML выполняется best-effort; после ошибки новая proxy-сессия восстанавливается."""
    await databases.seed(
        "remote", models.UserDescription(nick="alice", description="remote", guild_id=100)
    )
    async with models.async_session() as session:
        await session.execute(
            insert(models.UserDescription).values(nick="alice", description="local", guild_id=100)
        )
        await session.commit()
    for side, description in (("local", "local"), ("remote", "remote")):
        row, = await databases.rows(side, models.UserDescription)
        assert (row.nick, row.description) == ("alice", description)

    # The failed remote transaction is not reusable through the same proxy.  A
    # fresh application operation can still write to both databases.
    async with models.async_session() as session:
        await session.execute(
            insert(models.UserDescription).values(nick="bob", description="both", guild_id=100)
        )
        await session.commit()
    for side in ("local", "remote"):
        rows = {
            row.nick: row.description for row in await databases.rows(side, models.UserDescription)
        }
        expected = {"alice": side, "bob": "both"}
        assert rows == expected
    assert "Ошибка выполнения запроса в удаленной БД" in capsys.readouterr().out


async def test_local_integrity_failure_can_roll_back_both_sessions(databases: Databases) -> None:
    """Ошибка local до commit не оставляет незавершённую запись в remote."""
    await databases.seed(
        "local", models.UserDescription(nick="alice", description="old", guild_id=100)
    )
    async with models.async_session() as session:
        session.add(models.UserDescription(nick="alice", description="duplicate", guild_id=100))
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
        session.add(models.UserDescription(nick="bob", description="ok", guild_id=100))
        await session.commit()
    assert {row.nick for row in await databases.rows("local", models.UserDescription)} == {
        "alice", "bob"
    }
    assert [row.nick for row in await databases.rows("remote", models.UserDescription)] == ["bob"]