"""Регрессии зеркальных UPDATE через реальные функции сохранения описаний."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.data import decorators, emoji_descriptions_cache, models, request, user_descriptions_cache
from tests.integration.conftest import Databases

type SaveDescription = Callable[[str, str, int], Awaitable[str]]

DESCRIPTION_STORES = [
    pytest.param(
        emoji_descriptions_cache.save, models.GuildEmoji, "name", emoji_descriptions_cache,
        id="emoji",
    ),
    pytest.param(
        user_descriptions_cache.save, models.UserDescription, "nick", user_descriptions_cache,
        id="user-cache",
    ),
    pytest.param(
        request.save_user_description, models.UserDescription, "nick", None, id="user-request"
    ),
]


@pytest.mark.parametrize("save, model, key, cache", DESCRIPTION_STORES)
@pytest.mark.parametrize("local_description", ["old", "new"])
async def test_save_updates_business_key_despite_different_ids(
    databases: Databases,
    save: SaveDescription,
    model: type[models.Base],
    key: str,
    cache: ModuleType | None,
    local_description: str,
) -> None:
    """Даже неизменный local обновляет stale remote, не затрагивая чужие ID/guild."""
    for side, target_id, description in (
        ("local", 10, local_description), ("remote", 20, "stale")
    ):
        await databases.seed(
            side,
            model(id=target_id, guild_id=100, description=description, **{key: "same"}),
            model(id=30, guild_id=200, description="other guild", **{key: "same"}),
            model(id=40, guild_id=100, description="other key", **{key: "other"}),
        )
    await databases.seed(
        "remote", model(id=10, guild_id=100, description="unrelated", **{key: "unrelated"})
    )
    if cache is not None:
        await cache.load_all()

    # Второй вызов с тем же текстом не должен создавать дубликаты или пропускать UPDATE.
    for _ in range(2):
        assert "обновлено" in await save("same", "new", 100)
        for side, target_id in (("local", 10), ("remote", 20)):
            rows = await databases.rows(side, model)
            expected = {
                (target_id, 100, "same", "new"),
                (30, 200, "same", "other guild"),
                (40, 100, "other", "other key"),
            }
            if side == "remote":
                expected.add((10, 100, "unrelated", "unrelated"))
            assert len(rows) == len(expected)
            assert {
                (row.id, row.guild_id, getattr(row, key), row.description) for row in rows
            } == expected
        if cache is not None:
            assert cache.get(100) == {"same": "new", "other": "other key"}
            assert cache.get(200) == {"same": "other guild"}


@pytest.mark.parametrize("save, model, key, cache", DESCRIPTION_STORES)
async def test_update_does_not_backfill_missing_remote_row(
    databases: Databases,
    save: SaveDescription,
    model: type[models.Base],
    key: str,
    cache: ModuleType | None,
) -> None:
    """UPDATE сохраняет local, но не превращается в синхронизацию/INSERT в remote."""
    await databases.seed("local", model(guild_id=100, description="old", **{key: "same"}))
    assert "обновлено" in await save("same", "new", 100)
    row, = await databases.rows("local", model)
    assert row.description == "new"
    assert await databases.rows("remote", model) == []
    if cache is not None:
        assert cache.get(100) == {"same": "new"}


@pytest.mark.parametrize("save, model, key, cache", DESCRIPTION_STORES)
@pytest.mark.parametrize("side", ["local", "remote"])
@pytest.mark.parametrize("failure", ["execute", "commit"])
async def test_update_failure_preserves_local_first_contract_and_recovers(
    databases: Databases,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    save: SaveDescription,
    model: type[models.Base],
    key: str,
    cache: ModuleType | None,
    side: str,
    failure: str,
) -> None:
    """Local failure откатывает обе БД; remote failure не отменяет local и кэш."""
    await save("same", "old", 100)
    maker = getattr(databases, side)
    if failure == "execute":
        # Реальная ошибка UPDATE переводит транзакцию PostgreSQL в failed state.
        async with maker.begin() as session:
            await session.execute(text(
                f"ALTER TABLE {model.__tablename__} "
                "ADD CONSTRAINT reject_new CHECK (description <> 'new')"
            ))

    @asynccontextmanager
    async def failing_commit_session() -> AsyncIterator[models.DualSessionProxy]:
        async with models.async_session() as proxy:
            with monkeypatch.context() as patch:
                patch.setattr(
                    getattr(proxy, side), "commit",
                    AsyncMock(side_effect=SQLAlchemyError("injected commit failure")),
                )
                yield proxy

    with monkeypatch.context() as patch:
        if failure == "commit":
            patch.setattr(decorators, "async_session", failing_commit_session)
        if side == "local":
            with pytest.raises(RuntimeError, match="Ошибка базы данных"):
                await save("same", "new", 100)
        else:
            assert "обновлено" in await save("same", "new", 100)

    local_description = "old" if side == "local" else "new"
    for database, description in (("local", local_description), ("remote", "old")):
        row, = await databases.rows(database, model)
        assert (getattr(row, key), row.description) == ("same", description)
    if cache is not None:
        assert cache.get(100) == {"same": local_description}
    if side == "remote":
        diagnostic = "выполнения запроса" if failure == "execute" else "коммита"
        assert f"Ошибка {diagnostic} в удаленной БД" in capsys.readouterr().out
    assert all(engine.pool.checkedout() == 0 for engine in databases.engines)

    if failure == "execute":
        async with maker.begin() as session:
            await session.execute(text(
                f"ALTER TABLE {model.__tablename__} DROP CONSTRAINT reject_new"
            ))
    # После снятия сбоя новый save использует свежие сессии и обновляет обе БД.
    assert "обновлено" in await save("same", "new", 100)
    for database in ("local", "remote"):
        row, = await databases.rows(database, model)
        assert row.description == "new"
    if cache is not None:
        assert cache.get(100) == {"same": "new"}