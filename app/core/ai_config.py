"""Polza-only runtime и независимые client caches по гильдиям."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.core.config import (
    POLZA_API_URL,
    POLZA_PROVIDER,
    PROVIDERS,
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
) -> AISettings:
    """Разрешает настройки только из явного guild cache или env для non-guild API."""
    if settings is None:
        settings = (
            guild_ai_config_cache.get_required(guild_id)
            if guild_id is not None
            else get_ai_settings()
        )
    return validate_ai_settings(settings, require_tokens=require_tokens)


def get_provider_config(name: str | None = None, *, guild_id: int | None = None) -> dict[str, str]:
    """Возвращает внутренний Polza-конфиг для совместимых старых импортов.

    Другие provider names намеренно не принимаются. Этот helper не используется
    административным API и не печатает credentials.
    """
    if name not in (None, POLZA_PROVIDER):
        raise ValueError("Поддерживается только provider polza")
    settings = _settings_for(guild_id, None, require_tokens=False)
    if not settings.token or not settings.token.strip():
        raise ValueError("Отсутствует chat token")
    return {
        "api_key": settings.token or "",
        "base_url": settings.api_url,
    }


def get_embedding_config(
    provider: str | None = None,
    *,
    guild_id: int | None = None,
    settings: AISettings | None = None,
) -> dict[str, str]:
    """Возвращает отдельные embedding URL, token и model для runtime."""
    if provider not in (None, POLZA_PROVIDER):
        raise ValueError("Поддерживается только embedding provider polza")
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if not resolved.embedding_token or not resolved.embedding_token.strip():
        raise ValueError("Отсутствует embedding token")
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
        raise TypeError("Передавайте guild_id, а provider switching отключён")
    resolved = _settings_for(guild_id, settings, require_tokens=False)
    if not resolved.token or not resolved.token.strip():
        raise ValueError("Отсутствует chat token")
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
        raise ValueError("Отсутствует embedding token")
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
        raise ValueError("Отсутствует embedding token")
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


def get_chat_client(
    guild_id: int | None = None,
    *,
    settings: AISettings | None = None,
) -> AsyncOpenAI:
    """Явное имя chat client API."""
    return get_client(guild_id, settings=settings)


def clear_chat_client_cache() -> None:
    """Очищает только chat namespace."""
    _chat_clients.clear()


def clear_embedding_client_cache() -> None:
    """Очищает только embedding namespace."""
    _embedding_clients.clear()
    _embedding_models.clear()


def get_active_provider() -> str:
    """Возвращает единственный разрешённый provider."""
    return POLZA_PROVIDER


def set_active_provider(name: str) -> None:
    """Запрещает переключение provider; Polza остаётся единственным runtime."""
    if name != POLZA_PROVIDER:
        raise ValueError("Переключение provider отключено: поддерживается только polza")


def get_available_providers() -> list[str]:
    """Возвращает только Polza."""
    return [POLZA_PROVIDER]


def next_provider() -> str:
    """Legacy API без циклического выбора provider."""
    raise ValueError("Переключение provider отключено: поддерживается только polza")


def get_model(guild_id: int | None = None, *, settings: AISettings | None = None) -> str:
    """Возвращает chat model нужной гильдии."""
    return _settings_for(guild_id, settings, require_tokens=False).chat_model


def get_mini_model(guild_id: int | None = None, *, settings: AISettings | None = None) -> str:
    """Возвращает mini-model нужной гильдии."""
    return _settings_for(guild_id, settings, require_tokens=False).mini_model


def set_model(model: str, *, guild_id: int | None = None) -> None:
    """Не изменяет persistent config; оставлен только для legacy compatibility."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Chat model не может быть пустой")
    if guild_id is not None:
        raise ValueError("Изменение chat model выполняется через repository гильдии")


__all__ = [
    "POLZA_API_URL",
    "POLZA_PROVIDER",
    "PROVIDERS",
    "clear_chat_client_cache",
    "clear_client_caches",
    "clear_embedding_client_cache",
    "get_active_provider",
    "get_available_providers",
    "get_chat_client",
    "get_client",
    "get_embedding_client",
    "get_embedding_config",
    "get_embedding_model",
    "get_mini_model",
    "get_model",
    "get_provider_config",
    "invalidate_chat_client",
    "invalidate_embedding_client",
    "invalidate_embedding_runtime",
    "next_provider",
    "set_active_provider",
    "set_model",
]
