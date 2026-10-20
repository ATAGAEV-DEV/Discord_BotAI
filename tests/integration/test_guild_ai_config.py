"""Интеграционные проверки полной guild AI-конфигурации в БД."""

import pytest

from app.core.config import AISettings
from app.data import models
from app.data.repositories import guild_ai_config
from tests.integration.conftest import Databases


def _settings() -> AISettings:
    """Создаёт полные credentials для тестовой гильдии."""
    return AISettings(
        token="integration-chat-secret",
        embedding_token="integration-embedding-secret",
        api_url="https://chat.integration.example/v1",
        embedding_api_url="https://embedding.integration.example/v1",
        chat_model="integration-chat-model",
        mini_model="integration-mini-model",
        embedding_model="integration-embedding-model",
    )


@pytest.mark.integration
async def test_guild_ai_config_is_persisted_in_database(
    databases: Databases,
) -> None:
    """Новая строка сохраняется с двумя credentials в БД."""
    settings = _settings()

    from app.data.models import async_session

    async with async_session() as session:
        result = await guild_ai_config.ensure(session, 7001, settings)

    assert result.token == settings.token
    assert result.embedding_token == settings.embedding_token
    rows = await databases.rows(models.GuildAIConfig)
    assert len(rows) == 1
    row = rows[0]
    assert row.guild_id == 7001
    assert row.token == settings.token
    assert row.embedding_token == settings.embedding_token
    assert row.api_url == settings.api_url
    assert row.embedding_api_url == settings.embedding_api_url


@pytest.mark.integration
async def test_existing_row_is_not_overwritten_by_environment(
    databases: Databases,
) -> None:
    """Повторный ensure использует БД, а не новые env-кандидаты."""
    from app.data.models import async_session

    stored = _settings()
    candidate = AISettings(
        token="different-chat-secret",
        embedding_token="different-embedding-secret",
        api_url="https://other-chat.example/v1",
        embedding_api_url="https://other-embedding.example/v1",
    )
    async with async_session() as session:
        await guild_ai_config.ensure(session, 7002, stored)
        result = await guild_ai_config.ensure(session, 7002, candidate)

    assert result.token == stored.token
    assert result.embedding_token == stored.embedding_token
    assert len(await databases.rows(models.GuildAIConfig)) == 1

