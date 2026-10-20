"""Реальные PostgreSQL-типы, ограничения и семантика сессии."""

from datetime import datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.data import admins, emoji_descriptions_cache, models, user_descriptions_cache
from tests.integration.conftest import Databases


async def test_schema_is_empty_and_connections_are_isolated(databases: Databases) -> None:
    """Каждый тест начинает с пустых таблиц, кэшей и своей схемы."""
    async with databases.session_maker() as session:
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
    """Snowflake, вложенный JSONB, Unicode и серверные даты сохраняются."""
    snowflake = 9223372036854775000
    context = {"dialog": [{"text": "Привет 👋", "ok": True}], "empty": None}
    async with models.async_session() as session:
        session.add(models.User(user_id=snowflake, name="Тест", context=context))
        session.add(models.UserMessageStats(user_id=snowflake, guild_id=100, name="Тест"))
        await session.commit()
    user, = await databases.rows(models.User)
    stats, = await databases.rows(models.UserMessageStats)
    assert user.user_id == snowflake
    assert user.context == context
    assert isinstance(user.datetime_insert, datetime)
    assert stats.message_count == 0
    assert isinstance(stats.last_updated, datetime)
    async with databases.session_maker() as session:
        column_type = await session.scalar(text("SELECT pg_typeof(context)::text FROM users"))
        assert column_type == "jsonb"


async def test_unique_not_null_and_composite_constraints(databases: Databases) -> None:
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
        await databases.seed(model(guild_id=100, **values), model(guild_id=200, **values))
        async with databases.session_maker() as session:
            session.add(model(guild_id=100, **values))
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
            assert await session.scalar(select(func.count()).select_from(model)) == 2

