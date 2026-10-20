"""Регрессии UPDATE через реальные функции сохранения описаний."""

from collections.abc import Awaitable, Callable
from types import ModuleType

import pytest
from sqlalchemy import text

from app.data import emoji_descriptions_cache, models, request, user_descriptions_cache
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
async def test_save_updates_business_key_with_different_ids(
    databases: Databases,
    save: SaveDescription,
    model: type[models.Base],
    key: str,
    cache: ModuleType | None,
) -> None:
    """Повторный save обновляет описание, не затрагивая чужие ID/guild."""
    await databases.seed(
        model(id=10, guild_id=100, description="old", **{key: "same"}),
        model(id=30, guild_id=200, description="other guild", **{key: "same"}),
        model(id=40, guild_id=100, description="other key", **{key: "other"}),
    )
    if cache is not None:
        await cache.load_all()

    for _ in range(2):
        assert "обновлено" in await save("same", "new", 100)
        rows = await databases.rows(model)
        expected = {
            (10, 100, "same", "new"),
            (30, 200, "same", "other guild"),
            (40, 100, "other", "other key"),
        }
        assert len(rows) == len(expected)
        assert {
            (row.id, row.guild_id, getattr(row, key), row.description) for row in rows
        } == expected
        if cache is not None:
            assert cache.get(100) == {"same": "new", "other": "other key"}
            assert cache.get(200) == {"same": "other guild"}


@pytest.mark.parametrize("save, model, key, cache", DESCRIPTION_STORES)
async def test_update_does_not_backfill_missing_row(
    databases: Databases,
    save: SaveDescription,
    model: type[models.Base],
    key: str,
    cache: ModuleType | None,
) -> None:
    """UPDATE сохраняет строку; повторный вызов её обновляет."""
    await databases.seed(model(guild_id=100, description="old", **{key: "same"}))
    assert "обновлено" in await save("same", "new", 100)
    row, = await databases.rows(model)
    assert row.description == "new"
    if cache is not None:
        assert cache.get(100) == {"same": "new"}

