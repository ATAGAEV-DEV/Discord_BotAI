"""Тесты общих настроек приложения."""

import pytest

from app.core import config
from app.core.config import (
    AISettings,
    DatabaseSettings,
    SchedulerSettings,
    get_ai_settings,
    get_command_prefix,
    get_database_settings,
    get_scheduler_settings,
    validate_ai_settings,
)


def test_database_schema_defaults_to_discord(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если переменная не задана, используется прежняя схема discord."""
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)

    assert get_database_settings() == DatabaseSettings(schema="discord")


def test_database_schema_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Схема из окружения имеет приоритет над значением по умолчанию."""
    monkeypatch.setenv("DATABASE_SCHEMA", "staging")

    assert get_database_settings().schema == "staging"


def test_empty_database_schema_keeps_public_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустое значение остаётся пустым для перехода на public в models.py."""
    monkeypatch.setenv("DATABASE_SCHEMA", "")

    assert get_database_settings().schema == ""


def test_ai_settings_uses_chat_values_for_missing_embedding_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующие embedding provider и URL наследуются от chat-настроек."""
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
    assert settings.api_url == "https://chat.example/v1"
    assert settings.embedding_provider == settings.provider
    assert settings.embedding_api_url == settings.api_url


def test_ai_settings_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Значения AI из окружения имеют приоритет над значениями по умолчанию."""
    monkeypatch.setenv("AI_PROVIDER", "custom-chat-provider")
    monkeypatch.setenv("AI_API_URL", "https://chat.example/v1")
    monkeypatch.setenv("AI_MODEL", "custom-model")
    monkeypatch.setenv("AI_MODEL_MINI", "custom-mini-model")
    monkeypatch.setenv("AI_EMBEDDING_PROVIDER", "custom-embedding-provider")
    monkeypatch.setenv("AI_EMBEDDING_API_URL", "https://embedding.example/v1")
    monkeypatch.setenv("AI_EMBEDDING_MODEL", "custom-embedding-model")
    monkeypatch.setenv("AI_TOKEN", "chat-secret")
    monkeypatch.setenv("AI_EMBEDDING_TOKEN", "embedding-secret")

    assert get_ai_settings() == AISettings(
        provider="custom-chat-provider",
        api_url="https://chat.example/v1",
        token="chat-secret",
        model="custom-model",
        mini_model="custom-mini-model",
        embedding_provider="custom-embedding-provider",
        embedding_api_url="https://embedding.example/v1",
        embedding_token="embedding-secret",
        embedding_model="custom-embedding-model",
    )


def test_empty_chat_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без provider и URL конфигурация не получает vendor-specific fallback."""
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
        monkeypatch.setenv(name, "")

    with pytest.raises(ValueError, match="provider|api_url"):
        get_ai_settings()


def test_missing_models_are_kept_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Модели без env не получают vendor-specific fallback."""
    monkeypatch.setenv("AI_PROVIDER", "custom-chat-provider")
    monkeypatch.setenv("AI_API_URL", "https://chat.example/v1")
    for name in ("AI_MODEL", "AI_MODEL_MINI", "AI_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)

    settings = get_ai_settings()

    assert settings.chat_model is None
    assert settings.mini_model is None
    assert settings.embedding_model is None
    with pytest.raises(ValueError, match="chat_model"):
        validate_ai_settings(settings, require_tokens=False)


def test_scheduler_settings_defaults() -> None:
    """Значения планировщика по умолчанию заданы в Python-конфиге."""
    assert config.BOT_TIMEZONE == "Europe/Moscow"
    assert config.YOUTUBE_CHECK_INTERVAL_MINUTES == 5

    assert get_scheduler_settings() == SchedulerSettings(
        timezone="Europe/Moscow", youtube_check_interval_minutes=5
    )


def test_scheduler_settings_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Редактирование констант меняет настройки независимо от окружения."""
    monkeypatch.setattr(config, "BOT_TIMEZONE", "Asia/Tokyo")
    monkeypatch.setattr(config, "YOUTUBE_CHECK_INTERVAL_MINUTES", 12)
    monkeypatch.setenv("BOT_TIMEZONE", "Mars/Olympus")
    monkeypatch.setenv("YOUTUBE_CHECK_INTERVAL_MINUTES", "0")

    assert get_scheduler_settings() == SchedulerSettings(
        timezone="Asia/Tokyo", youtube_check_interval_minutes=12
    )


@pytest.mark.parametrize("timezone", ["", "Mars/Olympus"])
def test_invalid_scheduler_timezone(monkeypatch: pytest.MonkeyPatch, timezone: str) -> None:
    """Неизвестная или пустая таймзона не подменяется значением по умолчанию."""
    monkeypatch.setattr(config, "BOT_TIMEZONE", timezone)

    with pytest.raises(ValueError, match="BOT_TIMEZONE"):
        get_scheduler_settings()


@pytest.mark.parametrize("interval", ["", "abc", 1.5, 0, -1, True])
def test_invalid_scheduler_interval(monkeypatch: pytest.MonkeyPatch, interval: object) -> None:
    """Интервал должен быть положительным целым числом."""
    monkeypatch.setattr(config, "YOUTUBE_CHECK_INTERVAL_MINUTES", interval)

    with pytest.raises(ValueError, match="YOUTUBE_CHECK_INTERVAL_MINUTES"):
        get_scheduler_settings()


def test_command_prefix_defaults_to_exclamation() -> None:
    """Префикс по умолчанию хранится в Python-конфиге."""
    assert config.COMMAND_PREFIX == "!"

    assert get_command_prefix() == "!"


def test_bot_limits_defaults() -> None:
    """Лимиты при запуске сохраняют прежние значения."""
    assert config.CONTEXT_LIMIT == 100
    assert config.REPORT_MSG_LIMIT == 15
    assert config.REPORT_TIME_LIMIT == 60


def test_rag_and_history_defaults() -> None:
    """RAG и история канала сохраняют свои дефолтные значения."""
    assert config.RAG_MIN_RELEVANCE_SCORE == 0.20
    assert config.CONVERSATION_MAX_PAIRS == 20


def test_command_prefix_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Префикс берётся из конфига, даже если в окружении задано другое значение."""
    monkeypatch.setattr(config, "COMMAND_PREFIX", "??")
    monkeypatch.setenv("COMMAND_PREFIX", "")

    assert get_command_prefix() == "??"


@pytest.mark.parametrize("prefix", ["", "  "])
def test_empty_command_prefix_is_rejected(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Пустой префикс не должен превращать все сообщения в команды."""
    monkeypatch.setattr(config, "COMMAND_PREFIX", prefix)

    with pytest.raises(ValueError, match="COMMAND_PREFIX"):
        get_command_prefix()


def test_ai_settings_model_property_is_alias_for_chat_model() -> None:
    """AISettings.model — legacy alias для chat_model."""
    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        chat_model="gpt-4",
    )
    assert settings.model == "gpt-4"

    settings_no_model = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        chat_model=None,
    )
    assert settings_no_model.model is None


def test_validate_ai_settings_raises_type_error_for_non_string_url() -> None:
    """validate_ai_settings выбрасывает TypeError для не-строкового api_url."""
    from app.core.config import validate_ai_settings

    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
    )
    # Патчим api_url объектом, не строкой
    object.__setattr__(settings, "api_url", 12345)

    with pytest.raises(TypeError, match="api_url"):
        validate_ai_settings(settings, require_tokens=False, require_models=False)


def test_validate_ai_settings_raises_type_error_for_non_string_embedding_url() -> None:
    """validate_ai_settings выбрасывает TypeError для не-строкового embedding_api_url."""
    from app.core.config import validate_ai_settings

    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        embedding_api_url="https://p.example/v1",
    )
    object.__setattr__(settings, "embedding_api_url", None)

    with pytest.raises(TypeError, match="embedding_api_url"):
        validate_ai_settings(settings, require_tokens=False, require_models=False)


def test_validate_ai_settings_raises_type_error_for_non_string_model() -> None:
    """validate_ai_settings выбрасывает TypeError для не-строкового значения модели."""
    from app.core.config import validate_ai_settings

    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        chat_model="gpt-4",
        mini_model="gpt-4-mini",
        embedding_model="embed-v1",
    )
    # Помещаем не-строку, не None — должен сработать ветка TypeError
    object.__setattr__(settings, "chat_model", 999)

    with pytest.raises(TypeError, match="chat_model"):
        validate_ai_settings(settings, require_tokens=False)


def test_validate_ai_settings_raises_error_for_empty_model_string() -> None:
    """validate_ai_settings выбрасывает AIConfigurationError для пустой строки модели."""
    from app.core.config import AIConfigurationError, validate_ai_settings

    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        chat_model="   ",
        mini_model="gpt-4-mini",
        embedding_model="embed-v1",
    )

    with pytest.raises(AIConfigurationError, match="chat_model"):
        validate_ai_settings(settings, require_tokens=False)


def test_validate_ai_settings_raises_error_for_missing_chat_token() -> None:
    """validate_ai_settings выбрасывает AIConfigurationError при отсутствии chat token."""
    from app.core.config import AIConfigurationError, validate_ai_settings

    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        chat_model="gpt-4",
        mini_model="gpt-4-mini",
        embedding_model="embed-v1",
        token=None,
        embedding_token="emb-secret",
    )

    with pytest.raises(AIConfigurationError, match="chat token"):
        validate_ai_settings(settings)


def test_validate_ai_settings_raises_error_for_invalid_embedding_url() -> None:
    """validate_ai_settings выбрасывает AIConfigurationError для невалидного embedding URL."""
    from app.core.config import AIConfigurationError, validate_ai_settings

    settings = AISettings(
        provider="p",
        api_url="https://p.example/v1",
        embedding_api_url="not-a-valid-url",
    )

    with pytest.raises(AIConfigurationError, match="embedding_api_url"):
        validate_ai_settings(settings, require_tokens=False, require_models=False)
