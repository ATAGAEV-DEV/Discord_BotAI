"""Unit-тесты для app/core/ai_config.py."""

import pytest
from openai import AsyncOpenAI

from app.core import ai_config
from app.core.ai_config import (
    PROVIDERS,
    get_active_provider,
    get_available_providers,
    get_client,
    get_model,
    get_provider_config,
    next_provider,
    set_active_provider,
    set_model,
)
from app.core.config import PROVIDERS as CONFIG_PROVIDERS
from app.core.config import AISettings, get_ai_settings


@pytest.fixture(autouse=True)
def _reset_ai_config_state() -> None:
    """Сбрасывает глобальное состояние ai_config перед каждым тестом."""
    ai_config._active_provider = "aitunnel"
    ai_config._active_model = "claude-haiku-4.5"
    ai_config._cached_client = None
    ai_config._cached_provider_name = None


# ── get_provider_config ─────────────────────────────────────────


class TestGetProviderConfig:
    """Тесты для функции get_provider_config."""

    def test_returns_config_for_valid_provider(self) -> None:
        """Возвращает конфиг для существующего провайдера."""
        config = get_provider_config("proxyapi")
        assert "api_key" in config
        assert config["base_url"] == "https://api.proxyapi.ru/openai/v1"

    def test_returns_config_for_active_provider(self) -> None:
        """Без аргумента возвращает конфиг активного провайдера."""
        set_active_provider("polza")
        config = get_provider_config()
        assert config["base_url"] == "https://api.polza.ai/api/v1"

    def test_raises_on_unknown_provider(self) -> None:
        """Ошибка при неизвестном провайдере."""
        with pytest.raises(ValueError, match="Неизвестный провайдер"):
            get_provider_config("nonexistent")

    def test_registry_is_shared_with_config(self) -> None:
        """Старый импорт PROVIDERS продолжает работать после переноса реестра."""
        assert PROVIDERS is CONFIG_PROVIDERS


class TestAISettings:
    """Тесты для общих AI-настроек."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """При отсутствии переменных используются прежние значения по умолчанию."""
        for name in (
            "AI_PROVIDER",
            "AI_MODEL",
            "AI_MODEL_MINI",
            "AI_EMBEDDING_PROVIDER",
            "AI_EMBEDDING_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

        assert get_ai_settings() == AISettings(
            provider="polza",
            model="openai/gpt-6-luna",
            mini_model="openai/gpt-6-luna",
            embedding_provider="polza",
            embedding_model="text-embedding-3-large",
        )

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Все настройки переопределяются переменными окружения."""
        monkeypatch.setenv("AI_PROVIDER", "aitunnel")
        monkeypatch.setenv("AI_MODEL", "custom-chat")
        monkeypatch.setenv("AI_MODEL_MINI", "custom-mini")
        monkeypatch.setenv("AI_EMBEDDING_PROVIDER", "proxyapi")
        monkeypatch.setenv("AI_EMBEDDING_MODEL", "custom-embedding")

        assert get_ai_settings() == AISettings(
            provider="aitunnel",
            model="custom-chat",
            mini_model="custom-mini",
            embedding_provider="proxyapi",
            embedding_model="custom-embedding",
        )


class TestGetEmbeddingConfig:
    """Тесты для конфигурации embeddings."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Embeddings по умолчанию используют polza и прежнюю модель."""
        monkeypatch.delenv("AI_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("AI_EMBEDDING_MODEL", raising=False)

        config = ai_config.get_embedding_config()

        assert config["base_url"] == PROVIDERS["polza"]["base_url"]
        assert config["model"] == "text-embedding-3-large"

    def test_env_and_explicit_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Настройки окружения читаются при вызове, аргумент важнее провайдера из env."""
        monkeypatch.setenv("AI_EMBEDDING_PROVIDER", "aitunnel")
        monkeypatch.setenv("AI_EMBEDDING_MODEL", "custom-embedding")

        assert ai_config.get_embedding_config()["base_url"] == PROVIDERS["aitunnel"]["base_url"]
        config = ai_config.get_embedding_config("proxyapi")
        assert config["base_url"] == PROVIDERS["proxyapi"]["base_url"]
        assert config["model"] == "custom-embedding"


# ── set_active_provider / get_active_provider ───────────────────


class TestSetGetActiveProvider:
    """Тесты для set_active_provider и get_active_provider."""

    def test_sets_and_gets_provider(self) -> None:
        """Устанавливает и возвращает активного провайдера."""
        set_active_provider("proxyapi")
        assert get_active_provider() == "proxyapi"

    def test_resets_cache_on_switch(self) -> None:
        """При смене провайдера кеш клиента сбрасывается."""
        get_client()
        assert ai_config._cached_client is not None

        set_active_provider("polza")
        assert ai_config._cached_client is None
        assert ai_config._cached_provider_name is None

    def test_raises_on_unknown_provider(self) -> None:
        """Ошибка при попытке установить неизвестного провайдера."""
        with pytest.raises(ValueError, match="Неизвестный провайдер"):
            set_active_provider("unknown")


# ── next_provider ───────────────────────────────────────────────


class TestNextProvider:
    """Тесты для функции next_provider."""

    def test_cycles_to_next_provider(self) -> None:
        """Переключает на следующего провайдера."""
        providers_list = list(PROVIDERS.keys())
        start = get_active_provider()
        start_index = providers_list.index(start)
        expected = providers_list[(start_index + 1) % len(providers_list)]

        result = next_provider()
        assert result == expected
        assert get_active_provider() == expected

    def test_wraps_around(self) -> None:
        """Возвращается к первому провайдеру после последнего."""
        providers_list = list(PROVIDERS.keys())
        # Переключаем до последнего
        set_active_provider(providers_list[-1])

        result = next_provider()
        assert result == providers_list[0]


# ── get_available_providers ─────────────────────────────────────


class TestGetAvailableProviders:
    """Тесты для функции get_available_providers."""

    def test_returns_all_providers(self) -> None:
        """Возвращает список всех провайдеров."""
        result = get_available_providers()
        assert set(result) == set(PROVIDERS.keys())
        assert len(result) == len(PROVIDERS)

    def test_returns_list(self) -> None:
        """Возвращает list, а не dict_keys."""
        assert isinstance(get_available_providers(), list)


# ── get_model / set_model ───────────────────────────────────────


class TestGetSetModel:
    """Тесты для get_model и set_model."""

    def test_default_model(self) -> None:
        """Модель по умолчанию — 'claude-haiku-4.5'."""
        assert get_model() == "claude-haiku-4.5"

    def test_set_and_get_model(self) -> None:
        """Устанавливает и возвращает новую модель."""
        set_model("gpt-4o")
        assert get_model() == "gpt-4o"


# ── get_client ──────────────────────────────────────────────────


class TestGetClient:
    """Тесты для функции get_client."""

    def test_returns_async_openai_client(self) -> None:
        """Возвращает экземпляр AsyncOpenAI."""
        client = get_client()
        assert isinstance(client, AsyncOpenAI)

    def test_caches_client(self) -> None:
        """Повторный вызов возвращает тот же объект."""
        client1 = get_client()
        client2 = get_client()
        assert client1 is client2

    def test_recreates_on_different_provider(self) -> None:
        """Создаёт новый клиент при смене провайдера."""
        client1 = get_client()
        set_active_provider("polza")
        client2 = get_client()
        assert client1 is not client2

    def test_explicit_provider_argument(self) -> None:
        """Можно явно указать провайдера."""
        client = get_client("proxyapi")
        assert isinstance(client, AsyncOpenAI)
        assert ai_config._cached_provider_name == "proxyapi"


# ── get_mini_model ──────────────────────────────────────────────


class TestGetMiniModel:
    """Тесты для get_mini_model."""

    def test_default_mini_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Модель по умолчанию — 'openai/gpt-6-luna'."""
        monkeypatch.delenv("AI_MODEL_MINI", raising=False)
        assert ai_config.get_mini_model() == "openai/gpt-6-luna"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Можно переопределить через переменную окружения."""
        monkeypatch.setenv("AI_MODEL_MINI", "gpt-3.5-turbo")
        assert ai_config.get_mini_model() == "gpt-3.5-turbo"
