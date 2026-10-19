"""Идемпотентность явной миграции guild AI-конфигурации."""

import pytest

from app.data import models
from app.data.migrations import guild_ai_config
from tests.integration.conftest import Databases


@pytest.mark.integration
async def test_migration_is_idempotent_and_does_not_touch_chat_cache(
    databases: Databases,
) -> None:
    """Повторный migration не меняет данные и не требует AI_TOKEN."""
    result = await guild_ai_config.migrate(
        local=databases.engines[0],
        remote=databases.engines[1],
        migration_token="migration-embedding-secret",
        migration_url="https://embedding.integration.example/v1",
    )
    repeated = await guild_ai_config.migrate(
        local=databases.engines[0],
        remote=databases.engines[1],
        migration_token="migration-embedding-secret",
        migration_url="https://embedding.integration.example/v1",
    )

    assert result.guild_count == 0
    assert result.backfilled_guilds == ()
    assert repeated.guild_count == 0
    assert await databases.rows("local", models.GuildAIConfig) == []
    assert await databases.rows("remote", models.GuildAIConfig) == []
