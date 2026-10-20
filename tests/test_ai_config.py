"""Unit-тесты provider-agnostic runtime и независимых client caches."""

import pytest
from openai import AsyncOpenAI

from app.core import ai_config
from app.core.ai_config import (
    get_client,
    get_embedding_client,
    get_embedding_config,
    get_mini_model,
    get_model,
    get_provider_config,
    invalidate_chat_client,
    invalidate_embedding_runtime,
)
from app.core.config import AIConfigurationError, AISettings, get_ai_settings
from app.data import guild_ai_config_cache
from app.data.repositories.guild_ai_config import GuildAIConfigError


def _settings(
    *,
    token: str = "chat-secret",
    embedding_token: str = "embedding-secret",
    api_url: str = "https://chat.example/v1",
    embedding_api_url: str = "https://embedding.example/v1",
    chat_model: str | None = "chat-model",
    mini_model: str | None = "mini-model",
    embedding_model: str | None = "embedding-model",
) -> AISettings:
    """Создаёт полные настройки без чтения рабочей среды."""
    return AISettings(
        provider="custom-chat-provider",
        api_url=api_url,
        token=token,
        chat_model=chat_model,
        mini_model=mini_model,
        embedding_provider="custom-embedding-provider",
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


def test_environment_provider_and_url_are_used_for_missing_embedding_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding provider и URL наследуются от настроенных chat-полей."""
    for name in (
        "AI_MODEL",
        "AI_MODEL_MINI",
        "AI_EMBEDDING_PROVIDER",
        "AI_EMBEDDING_API_URL",
        "AI_EMBEDDING_MODEL",
        "AI_TOKEN",
        "AI_EMBEDDING_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_PROVIDER", "custom-chat-provider")
    monkeypatch.setenv("AI_API_URL", "https://chat.example/v1")

    settings = get_ai_settings()

    assert settings.provider == "custom-chat-provider"
    assert settings.embedding_provider == "custom-chat-provider"
    assert settings.api_url == "https://chat.example/v1"
    assert settings.embedding_api_url == "https://chat.example/v1"
    assert settings.token is None
    assert settings.embedding_token is None


def test_environment_values_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Каждый env credential и URL попадает только в своё поле."""
    monkeypatch.setenv("AI_API_URL", "https://chat.example/v2")
    monkeypatch.setenv("AI_EMBEDDING_API_URL", "https://embed.example/v3")
    monkeypatch.setenv("AI_PROVIDER", "custom-chat-provider")
    monkeypatch.setenv("AI_EMBEDDING_PROVIDER", "custom-embedding-provider")
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


def test_provider_helpers_accept_arbitrary_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime helpers не ограничивают provider names фиксированным списком."""
    monkeypatch.setenv("AI_PROVIDER", "chat-provider.example")
    monkeypatch.setenv("AI_API_URL", "https://chat-provider.example/v1")
    monkeypatch.setenv("AI_TOKEN", "chat-secret")

    chat_config = get_provider_config("another-chat-provider")
    embedding_config = get_embedding_config(
        "another-embedding-provider",
        settings=_settings(),
    )

    assert chat_config == {
        "api_key": "chat-secret",
        "base_url": "https://chat-provider.example/v1",
    }
    assert embedding_config["base_url"] == "https://embedding.example/v1"


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


def test_model_lookup_reports_missing_model() -> None:
    """Runtime не вызывает API, если модель гильдии не настроена."""
    guild_ai_config_cache._cache[999] = _settings(chat_model=None)

    with pytest.raises(ValueError, match="chat_model"):
        get_model(999)

    with pytest.raises(ValueError, match="mini_model"):
        get_mini_model(999, settings=_settings(chat_model="chat-model", mini_model=None))


def test_embedding_config_reports_missing_embedding_model() -> None:
    """Embedding runtime не получает None в поле model."""
    with pytest.raises(ValueError, match="embedding_model"):
        get_embedding_config(settings=_settings(embedding_model=None))


def test_embedding_config_keeps_embedding_url_and_secret_internal() -> None:
    """Embedding config использует собственные поля, а repr не раскрывает token."""
    settings = _settings()
    config = get_embedding_config(settings=settings)

    assert config["base_url"] == settings.embedding_api_url
    assert config["api_key"] == settings.embedding_token
    assert settings.token not in repr(settings)
    assert settings.embedding_token not in repr(settings)
    assert "token" not in settings.public_dict()


# ---------------------------------------------------------------------------
# get_provider_config (строки 58-63)
# ---------------------------------------------------------------------------

def test_get_provider_config_returns_chat_credentials() -> None:
    """get_provider_config возвращает chat api_key и base_url."""
    settings = _settings()
    guild_ai_config_cache._cache[10] = settings

    config = get_provider_config(guild_id=10)

    assert config["api_key"] == settings.token
    assert config["base_url"] == settings.api_url


def test_get_provider_config_raises_without_token() -> None:
    """get_provider_config выбрасывает AIConfigurationError если chat token отсутствует."""
    from app.core.config import AIConfigurationError

    settings = _settings(token=None)
    guild_ai_config_cache._cache[11] = settings

    with pytest.raises(AIConfigurationError, match="chat token"):
        get_provider_config(guild_id=11)


def test_get_provider_config_ignores_name_parameter() -> None:
    """Legacy параметр name игнорируется — используются текущие настройки."""
    settings = _settings()
    guild_ai_config_cache._cache[12] = settings

    config = get_provider_config(name="ignored-name", guild_id=12)

    assert config["base_url"] == settings.api_url


# ---------------------------------------------------------------------------
# get_embedding_config — missing token (строка 76)
# ---------------------------------------------------------------------------

def test_get_embedding_config_raises_without_embedding_token() -> None:
    """get_embedding_config выбрасывает AIConfigurationError если embedding token отсутствует."""
    from app.core.config import AIConfigurationError

    with pytest.raises(AIConfigurationError, match="embedding token"):
        get_embedding_config(settings=_settings(embedding_token=None))


# ---------------------------------------------------------------------------
# get_client — строки 93, 96
# ---------------------------------------------------------------------------

def test_get_client_raises_type_error_for_string_guild_id() -> None:
    """get_client выбрасывает TypeError если guild_id — строка."""
    with pytest.raises(TypeError, match="guild_id"):
        get_client("not-an-int")  # type: ignore[arg-type]


def test_get_client_raises_for_missing_token() -> None:
    """get_client выбрасывает AIConfigurationError если chat token отсутствует."""
    from app.core.config import AIConfigurationError

    with pytest.raises(AIConfigurationError, match="chat token"):
        get_client(settings=_settings(token=None))


# ---------------------------------------------------------------------------
# Legacy API: get_chat_client, clear_chat_client_cache,
#             clear_embedding_client_cache (строки 198, 203, 208-209)
# ---------------------------------------------------------------------------

def test_get_chat_client_delegates_to_get_client() -> None:
    """get_chat_client возвращает тот же объект, что и get_client."""
    from app.core.ai_config import get_chat_client

    settings = _settings()
    client_via_get = get_client(settings=settings)
    client_via_chat = get_chat_client(settings=settings)

    assert client_via_get is client_via_chat


def test_clear_chat_client_cache_clears_only_chat() -> None:
    """clear_chat_client_cache очищает только chat namespace."""
    from app.core import ai_config as ai_config_module
    from app.core.ai_config import clear_chat_client_cache

    settings = _settings()
    get_client(settings=settings)
    get_embedding_client(settings=settings)

    clear_chat_client_cache()

    assert len(ai_config_module._chat_clients) == 0
    assert len(ai_config_module._embedding_clients) == 1


def test_clear_embedding_client_cache_clears_only_embedding() -> None:
    """clear_embedding_client_cache очищает только embedding namespace."""
    from app.core import ai_config as ai_config_module
    from app.core.ai_config import clear_embedding_client_cache

    settings = _settings()
    get_client(settings=settings)
    get_embedding_client(settings=settings)
    # Вставляем fake объект напрямую в _embedding_models, минуя OpenAIEmbedding
    fake_key = (None, settings.embedding_api_url, settings.embedding_token or "", settings.embedding_model or "")
    ai_config_module._embedding_models[fake_key] = object()  # type: ignore[assignment]

    clear_embedding_client_cache()

    assert len(ai_config_module._chat_clients) == 1
    assert len(ai_config_module._embedding_clients) == 0
    assert len(ai_config_module._embedding_models) == 0


# ---------------------------------------------------------------------------
# get_active_provider, get_available_providers, next_provider,
# set_active_provider (строки 214, 223-230, 235)
# ---------------------------------------------------------------------------

def test_get_active_provider_returns_provider_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_active_provider возвращает provider из текущих настроек окружения."""
    from app.core.ai_config import get_active_provider

    monkeypatch.setenv("AI_PROVIDER", "test-provider")
    monkeypatch.setenv("AI_API_URL", "https://test.example/v1")

    assert get_active_provider() == "test-provider"


def test_get_available_providers_returns_list_with_active_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_available_providers возвращает список с одним текущим провайдером."""
    from app.core.ai_config import get_available_providers

    monkeypatch.setenv("AI_PROVIDER", "my-provider")
    monkeypatch.setenv("AI_API_URL", "https://my.example/v1")

    assert get_available_providers() == ["my-provider"]


def test_next_provider_returns_active_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """next_provider возвращает текущий active provider."""
    from app.core.ai_config import next_provider

    monkeypatch.setenv("AI_PROVIDER", "next-provider")
    monkeypatch.setenv("AI_API_URL", "https://next.example/v1")

    assert next_provider() == "next-provider"


def test_set_active_provider_accepts_valid_name() -> None:
    """set_active_provider принимает непустую строку без исключений."""
    from app.core.ai_config import set_active_provider

    set_active_provider("openai")  # не должно бросать


def test_set_active_provider_rejects_empty_name() -> None:
    """set_active_provider выбрасывает ValueError для пустой строки."""
    from app.core.ai_config import set_active_provider

    with pytest.raises(ValueError, match="Provider"):
        set_active_provider("   ")


def test_set_active_provider_rejects_non_string() -> None:
    """set_active_provider выбрасывает ValueError для не-строки."""
    from app.core.ai_config import set_active_provider

    with pytest.raises(ValueError, match="Provider"):
        set_active_provider(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# set_model (строки 256-259)
# ---------------------------------------------------------------------------

def test_set_model_accepts_valid_model_without_guild() -> None:
    """set_model принимает строку без guild_id без исключений."""
    from app.core.ai_config import set_model

    set_model("gpt-4")  # не должно бросать


def test_set_model_rejects_empty_string() -> None:
    """set_model выбрасывает ValueError для пустой строки."""
    from app.core.ai_config import set_model

    with pytest.raises(ValueError, match="Chat model"):
        set_model("  ")


def test_set_model_rejects_guild_id() -> None:
    """set_model выбрасывает ValueError если передан guild_id."""
    from app.core.ai_config import set_model

    with pytest.raises(ValueError, match="repository"):
        set_model("gpt-4", guild_id=42)



# ---------------------------------------------------------------------------
# get_model — happy path (строка 243)
# ---------------------------------------------------------------------------

def test_get_model_returns_chat_model_from_cache() -> None:
    """get_model возвращает chat_model из кэша гильдии."""
    guild_ai_config_cache._cache[555] = _settings(chat_model="gpt-4-turbo")

    result = get_model(555)

    assert result == "gpt-4-turbo"


# ---------------------------------------------------------------------------
# get_embedding_model — missing token (строка 139), missing model (строка 141)
# ---------------------------------------------------------------------------

def test_get_embedding_model_raises_when_token_is_empty() -> None:
    """get_embedding_model выбрасывает AIConfigurationError при пустом token."""
    from app.core.ai_config import get_embedding_model

    settings = _settings(embedding_token="")

    with pytest.raises(AIConfigurationError, match="embedding token"):
        get_embedding_model(settings=settings)


def test_get_embedding_model_raises_when_token_is_whitespace() -> None:
    """get_embedding_model выбрасывает AIConfigurationError при whitespace token."""
    from app.core.ai_config import get_embedding_model

    settings = _settings(embedding_token="   ")

    with pytest.raises(AIConfigurationError, match="embedding token"):
        get_embedding_model(settings=settings)


def test_get_embedding_model_raises_when_model_is_none() -> None:
    """get_embedding_model выбрасывает AIConfigurationError когда model=None."""
    from app.core.ai_config import get_embedding_model

    settings = _settings(embedding_model=None)

    with pytest.raises(AIConfigurationError, match="embedding_model"):
        get_embedding_model(settings=settings)




def test_invalidate_embedding_client_removes_matching_keys() -> None:
    """invalidate_embedding_client удаляет только ключи указанной гильдии."""
    from app.core import ai_config as ai_config_module
    from app.core.ai_config import invalidate_embedding_client

    settings = _settings()
    get_embedding_client(settings=settings)
    other_settings = _settings(embedding_token="other-secret")
    get_embedding_client(100, settings=other_settings)

    assert len(ai_config_module._embedding_clients) == 2

    invalidate_embedding_client(None)

    assert len(ai_config_module._embedding_clients) == 1
    remaining_key = list(ai_config_module._embedding_clients.keys())[0]
    assert remaining_key[0] == 100


# ---------------------------------------------------------------------------
# get_embedding_model — cache eviction old key (строки 152-153)
# ---------------------------------------------------------------------------

def test_get_embedding_model_evicts_old_key_for_same_guild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_embedding_model удаляет старый ключ гильдии при изменении модели."""
    from app.core import ai_config as ai_config_module
    from app.core.ai_config import get_embedding_model

    fake_embed = object()
    fake_class = type("FakeEmbedding", (), {"__init__": lambda s, **kw: None})
    monkeypatch.setattr("app.core.ai_config.OpenAIEmbedding", fake_class, raising=False)

    # Вставляем старый ключ для guild_id=50
    old_key = (50, "https://old.example/v1", "old-token", "old-model")
    ai_config_module._embedding_models[old_key] = fake_embed  # type: ignore[assignment]

    settings_new = _settings(
        embedding_token="new-token",
        embedding_model="text-embedding-ada-002",
        embedding_api_url="https://new.example/v1",
    )

    try:
        get_embedding_model(50, settings=settings_new)
    except Exception:
        pass  # OpenAIEmbedding может упасть — важно что старый ключ удалён

    assert old_key not in ai_config_module._embedding_models


# ---------------------------------------------------------------------------
# invalidate_embedding_runtime — loop over _embedding_models (строки 181-182)
# ---------------------------------------------------------------------------

def test_invalidate_embedding_runtime_removes_model_cache() -> None:
    """invalidate_embedding_runtime удаляет embedding model cache для гильдии."""
    from app.core import ai_config as ai_config_module
    from app.core.ai_config import invalidate_embedding_runtime

    settings = _settings()
    get_embedding_client(settings=settings)

    fake_key_same = (None, settings.embedding_api_url, settings.embedding_token or "", "m1")
    fake_key_other = (100, settings.embedding_api_url, settings.embedding_token or "", "m1")
    ai_config_module._embedding_models[fake_key_same] = object()  # type: ignore[assignment]
    ai_config_module._embedding_models[fake_key_other] = object()  # type: ignore[assignment]

    invalidate_embedding_runtime(None)

    assert fake_key_same not in ai_config_module._embedding_models
    assert fake_key_other in ai_config_module._embedding_models


# ---------------------------------------------------------------------------
# get_mini_model — return statement (строка 251)
# ---------------------------------------------------------------------------

def test_get_mini_model_returns_mini_model_from_settings() -> None:
    """get_mini_model возвращает mini_model из настроек."""
    settings = _settings(mini_model="gpt-4o-mini")
    guild_ai_config_cache._cache[77] = settings

    result = get_mini_model(77)

    assert result == "gpt-4o-mini"
