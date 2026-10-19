"""Unit-тесты Polza runtime и независимых client caches."""

import pytest
from openai import AsyncOpenAI

from app.core import ai_config
from app.core.ai_config import (
    get_client,
    get_embedding_client,
    get_embedding_config,
    get_model,
    invalidate_chat_client,
    invalidate_embedding_runtime,
)
from app.core.config import AISettings, get_ai_settings, validate_ai_settings
from app.data import guild_ai_config_cache
from app.data.repositories.guild_ai_config import GuildAIConfigError


def _settings(
    *,
    token: str = "chat-secret",
    embedding_token: str = "embedding-secret",
    api_url: str = "https://chat.example/v1",
    embedding_api_url: str = "https://embedding.example/v1",
    chat_model: str = "chat-model",
    embedding_model: str = "embedding-model",
) -> AISettings:
    """Создаёт полные настройки без чтения рабочей среды."""
    return AISettings(
        api_url=api_url,
        token=token,
        chat_model=chat_model,
        embedding_api_url=embedding_api_url,
        embedding_token=embedding_token,
        embedding_model=embedding_model,
    )


@pytest.fixture(autouse=True)
def _reset_runtime_cache() -> None:
    """Изолирует client и guild caches между тестами."""
    ai_config.clear_client_caches()
    guild_ai_config_cache.clear()
    yield
    ai_config.clear_client_caches()
    guild_ai_config_cache.clear()


def test_defaults_are_polza_and_have_separate_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Дефолты содержат Polza и два независимых URL."""
    for name in (
        "AI_PROVIDER",
        "AI_API_URL",
        "AI_MODEL",
        "AI_MODEL_MINI",
        "AI_EMBEDDING_PROVIDER",
        "AI_EMBEDDING_API_URL",
        "AI_EMBEDDING_MODEL",
        "AI_TOKEN",
        "AI_EMBEDDING_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_ai_settings()

    assert settings.provider == "polza"
    assert settings.embedding_provider == "polza"
    assert settings.api_url == "https://api.polza.ai/api/v1"
    assert settings.embedding_api_url == "https://api.polza.ai/api/v1"
    assert settings.token is None
    assert settings.embedding_token is None


def test_environment_values_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Каждый env credential и URL попадает только в своё поле."""
    monkeypatch.setenv("AI_API_URL", "https://chat.example/v2")
    monkeypatch.setenv("AI_EMBEDDING_API_URL", "https://embed.example/v3")
    monkeypatch.setenv("AI_TOKEN", "chat-secret")
    monkeypatch.setenv("AI_EMBEDDING_TOKEN", "embedding-secret")
    monkeypatch.setenv("AI_MODEL", "chat-model")
    monkeypatch.setenv("AI_MODEL_MINI", "mini-model")
    monkeypatch.setenv("AI_EMBEDDING_MODEL", "embed-model")

    settings = get_ai_settings()

    assert settings.api_url == "https://chat.example/v2"
    assert settings.embedding_api_url == "https://embed.example/v3"
    assert settings.token == "chat-secret"
    assert settings.embedding_token == "embedding-secret"
    assert settings.chat_model == "chat-model"
    assert settings.mini_model == "mini-model"
    assert settings.embedding_model == "embed-model"


@pytest.mark.parametrize("field", ["provider", "embedding_provider"])
def test_non_polza_provider_is_rejected(field: str) -> None:
    """Runtime не принимает провайдер, отличный от Polza."""
    settings = _settings()
    object.__setattr__(settings, field, "other")
    with pytest.raises(ValueError, match="provider"):
        validate_ai_settings(settings)


def test_embedding_token_never_falls_back_to_chat_token() -> None:
    """Отсутствующий embedding token не подменяется chat token."""
    settings = _settings(embedding_token=None)
    guild_ai_config_cache._cache[100] = settings

    assert get_client(100) is not None
    with pytest.raises(ValueError, match="embedding token"):
        get_embedding_client(100)


def test_chat_and_embedding_clients_are_isolated_by_namespace() -> None:
    """URL/token/guild изменения затрагивают только соответствующий cache."""
    settings = _settings()
    chat = get_client(100, settings=settings)
    assert isinstance(chat, AsyncOpenAI)
    assert get_client(100, settings=settings) is chat
    assert get_client(200, settings=settings) is not chat

    embedding = get_embedding_client(100, settings=settings)
    assert get_embedding_client(100, settings=settings) is embedding
    assert embedding is not chat

    changed_chat = _settings(token="new-chat-secret", embedding_token="embedding-secret")
    assert get_client(100, settings=changed_chat) is not chat
    assert get_embedding_client(100, settings=changed_chat) is embedding

    changed_embedding = _settings(token="chat-secret", embedding_token="new-embedding-secret")
    assert get_client(100, settings=changed_embedding) is chat
    assert get_embedding_client(100, settings=changed_embedding) is not embedding


def test_invalidation_is_independent() -> None:
    """Явная invalidation одного назначения не очищает другое."""
    settings = _settings()
    chat = get_client(1, settings=settings)
    embedding = get_embedding_client(1, settings=settings)

    invalidate_chat_client(1)
    assert get_client(1, settings=settings) is not chat
    assert get_embedding_client(1, settings=settings) is embedding

    invalidate_embedding_runtime(1)
    assert get_client(1, settings=settings) is not chat
    assert get_embedding_client(1, settings=settings) is not embedding


def test_model_lookup_requires_loaded_guild_configuration() -> None:
    """Guild runtime не использует env при cache miss."""
    with pytest.raises(GuildAIConfigError, match="не загружена"):
        get_model(999)


def test_embedding_config_keeps_embedding_url_and_secret_internal() -> None:
    """Embedding config использует собственные поля, а repr не раскрывает token."""
    settings = _settings()
    config = get_embedding_config(settings=settings)

    assert config["base_url"] == settings.embedding_api_url
    assert config["api_key"] == settings.embedding_token
    assert settings.token not in repr(settings)
    assert settings.embedding_token not in repr(settings)
    assert "token" not in settings.public_dict()
