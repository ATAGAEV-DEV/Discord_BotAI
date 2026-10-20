"""Provider-agnostic runtime и независимые client caches по гильдиям."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.core.config import (
    AIConfigurationError,
    AISettings,
    get_ai_settings,
    validate_ai_settings,
)
from app.data import guild_ai_config_cache

if TYPE_CHECKING:
    from llama_index.embeddings.openai import OpenAIEmbedding


ChatClientKey = tuple[int | None, str, str]
EmbeddingClientKey = tuple[int | None, str, str]

_chat_clients: dict[ChatClientKey, AsyncOpenAI] = {}
_embedding_clients: dict[EmbeddingClientKey, AsyncOpenAI] = {}
_embedding_models: dict[tuple[int | None, str, str, str], OpenAIEmbedding] = {}


def _settings_for(
    guild_id: int | None,
    settings: AISettings | None,
    *,
    require_tokens: bool,
    require_models: bool = False,
) -> AISettings:
    """Разрешает настройки только из явного guild cache или env для non-guild API."""
    if settings is None:
        settings = (
            guild_ai_config_cache.get_required(guild_id)
            if guild_id is not None
            else get_ai_settings()
        )
    return validate_ai_settings(
        settings,
        require_tokens=require_tokens,
        require_models=require_models,
    )



def get_embedding_config(
    provider: str | None = None,
    *,
    guild_id: int | None = None,
    settings: AISettings | None = None,
) -> dict[str, str]:
    """Возвращает отдельные embedding URL, token и model для runtime."""
    del provider
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if not resolved.embedding_token or not resolved.embedding_token.strip():
        raise AIConfigurationError("Отсутствует embedding token")
    if resolved.embedding_model is None:
        raise AIConfigurationError("Модель embedding_model не задана")
    return {
        "api_key": resolved.embedding_token or "",
        "base_url": resolved.embedding_api_url,
        "model": resolved.embedding_model,
    }


def get_client(
    guild_id: int | None = None,
    *,
    settings: AISettings | None = None,
) -> AsyncOpenAI:
    """Возвращает chat client по ключу ``(guild_id, api_url, token)``."""
    if isinstance(guild_id, str):
        raise TypeError("guild_id должен быть целым числом или None")
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if not resolved.token or not resolved.token.strip():
        raise AIConfigurationError("Отсутствует chat token")
    key: ChatClientKey = (guild_id, resolved.api_url, resolved.token or "")
    client = _chat_clients.get(key)
    if client is None:
        client = AsyncOpenAI(api_key=resolved.token or "", base_url=resolved.api_url)
        _chat_clients[key] = client
    return client


def get_embedding_client(
    guild_id: int | None = None,
    *,
    settings: AISettings | None = None,
) -> AsyncOpenAI:
    """Возвращает embedding client по независимому embedding key."""
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if not resolved.embedding_token or not resolved.embedding_token.strip():
        raise AIConfigurationError("Отсутствует embedding token")
    key: EmbeddingClientKey = (
        guild_id,
        resolved.embedding_api_url,
        resolved.embedding_token or "",
    )
    client = _embedding_clients.get(key)
    if client is None:
        client = AsyncOpenAI(
            api_key=resolved.embedding_token or "",
            base_url=resolved.embedding_api_url,
        )
        _embedding_clients[key] = client
    return client


def get_embedding_model(
    guild_id: int | None = None,
    *,
    settings: AISettings | None = None,
) -> OpenAIEmbedding:
    """Возвращает embedding model по URL, token, model и guild namespace."""
    from llama_index.embeddings.openai import OpenAIEmbedding

    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if not resolved.embedding_token or not resolved.embedding_token.strip():
        raise AIConfigurationError("Отсутствует embedding token")
    if resolved.embedding_model is None:
        raise AIConfigurationError("Модель embedding_model не задана")
    get_embedding_client(guild_id, settings=resolved)
    key = (
        guild_id,
        resolved.embedding_api_url,
        resolved.embedding_token or "",
        resolved.embedding_model,
    )
    model = _embedding_models.get(key)
    if model is None:
        for old_key in list(_embedding_models):
            if old_key[0] == guild_id:
                _embedding_models.pop(old_key, None)
        model = OpenAIEmbedding(
            api_key=resolved.embedding_token or "",
            api_base=resolved.embedding_api_url,
            model=resolved.embedding_model,
        )
        _embedding_models[key] = model
    return model


def invalidate_chat_client(guild_id: int) -> None:
    """Инвалидирует только chat clients указанной гильдии."""
    for key in list(_chat_clients):
        if key[0] == guild_id:
            _chat_clients.pop(key, None)


def invalidate_embedding_client(guild_id: int) -> None:
    """Инвалидирует только embedding clients указанной гильдии."""
    for key in list(_embedding_clients):
        if key[0] == guild_id:
            _embedding_clients.pop(key, None)


def invalidate_embedding_runtime(guild_id: int) -> None:
    """Инвалидирует embedding client и model cache, не затрагивая chat."""
    invalidate_embedding_client(guild_id)
    for key in list(_embedding_models):
        if key[0] == guild_id:
            _embedding_models.pop(key, None)


def clear_client_caches() -> None:
    """Очищает оба namespace; credentials не выводятся."""
    _chat_clients.clear()
    _embedding_clients.clear()
    _embedding_models.clear()



def get_active_provider() -> str:
    """Возвращает provider из текущей конфигурации."""
    return get_ai_settings().provider



def get_model(guild_id: int | None = None, *, settings: AISettings | None = None) -> str:
    """Возвращает chat model нужной гильдии."""
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if resolved.chat_model is None:
        raise AIConfigurationError("Модель chat_model не задана")
    return resolved.chat_model


def get_mini_model(guild_id: int | None = None, *, settings: AISettings | None = None) -> str:
    """Возвращает mini-model нужной гильдии."""
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if resolved.mini_model is None:
        raise AIConfigurationError("Модель mini_model не задана")
    return resolved.mini_model



__all__ = [
    "clear_client_caches",
    "get_active_provider",
    "get_client",
    "get_embedding_client",
    "get_embedding_config",
    "get_embedding_model",
    "get_mini_model",
    "get_model",
    "invalidate_chat_client",
    "invalidate_embedding_client",
    "invalidate_embedding_runtime",
]
