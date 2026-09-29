import os

from openai import AsyncOpenAI

from app.core.config import PROVIDERS, get_ai_settings

_initial_settings = get_ai_settings()
_active_provider: str = _initial_settings.provider
_active_model: str = _initial_settings.model
_cached_client: AsyncOpenAI | None = None
_cached_provider_name: str | None = None


def get_provider_config(name: str | None = None) -> dict[str, str]:
    """Возвращает конфиг провайдера по имени (или активного)."""
    provider_name = name or _active_provider
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Неизвестный провайдер: '{provider_name}'. Доступные: {', '.join(PROVIDERS.keys())}"
        )
    config = PROVIDERS[provider_name]
    return {
        "api_key": os.getenv(config["token_env"], ""),
        "base_url": config["base_url"],
    }


def get_embedding_config(provider: str | None = None) -> dict[str, str]:
    """Возвращает отдельный конфиг провайдера и модели embeddings."""
    settings = get_ai_settings()
    provider_name = provider or settings.embedding_provider
    config = get_provider_config(provider_name)
    return {
        "api_key": config["api_key"],
        "base_url": config["base_url"],
        "model": settings.embedding_model,
    }


def get_client(provider: str | None = None) -> AsyncOpenAI:
    """Возвращает клиент AsyncOpenAI для указанного провайдера.

    При повторном вызове с тем же провайдером возвращает
    кешированный клиент. Пересоздаёт при смене провайдера.
    """
    global _cached_client, _cached_provider_name  # noqa: PLW0603

    target = provider or _active_provider

    if _cached_client is not None and _cached_provider_name == target:
        return _cached_client

    config = get_provider_config(target)
    _cached_client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )
    _cached_provider_name = target
    print(f"[AI Config] Провайдер: {target} | URL: {config['base_url']}")
    return _cached_client


def set_active_provider(name: str) -> None:
    """Переключает активного провайдера.

    Сбрасывает кеш клиента, чтобы при следующем вызове
    get_client() создался новый клиент.
    """
    global _active_provider, _cached_client, _cached_provider_name  # noqa: PLW0603

    if name not in PROVIDERS:
        raise ValueError(
            f"Неизвестный провайдер: '{name}'. Доступные: {', '.join(PROVIDERS.keys())}"
        )
    _active_provider = name
    _cached_client = None
    _cached_provider_name = None


def get_active_provider() -> str:
    """Возвращает имя текущего активного провайдера."""
    return _active_provider


def get_model() -> str:
    """Возвращает текущую модель."""
    return _active_model


def set_model(model: str) -> None:
    """Устанавливает модель для генерации."""
    global _active_model  # noqa: PLW0603
    _active_model = model


def get_mini_model() -> str:
    """Возвращает текущую мини-модель (для простых задач)."""
    return get_ai_settings().mini_model


def next_provider() -> str:
    """Переключает на следующего провайдера по кругу и возвращает его имя."""
    providers_list = list(PROVIDERS.keys())
    current_index = providers_list.index(_active_provider)
    next_index = (current_index + 1) % len(providers_list)
    next_name = providers_list[next_index]
    set_active_provider(next_name)
    return next_name


def get_available_providers() -> list[str]:
    """Возвращает список доступных провайдеров."""
    return list(PROVIDERS.keys())
