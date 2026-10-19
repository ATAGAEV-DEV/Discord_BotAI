import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

import pytz
from dotenv import load_dotenv

load_dotenv(override=False)

AI_GENERATE_TIMEOUT: float = 60.0  # Секунды для генерации ответа и запроса к AI API
MAX_MESSAGE_LENGTH: int = 1000  # Максимальная длина входящего сообщения для отчетов
REPORT_IGNORE_PREFIX: str = "?"  # Сообщения с этим префиксом не входят в отчёты
COMMAND_PREFIX: str = "!"  # Префикс команд Discord-бота
BOT_TIMEZONE: str = "Europe/Moscow"  # Таймзона планировщика и логов YouTube
YOUTUBE_CHECK_INTERVAL_MINUTES: int = 5  # Частота проверки новых видео в минутах
CONTEXT_LIMIT: int = 100  # Количество строк контекста для RAG
REPORT_MSG_LIMIT: int = 15  # Порог сообщений для создания отчёта
REPORT_TIME_LIMIT: int = 60  # Время ожидания в минутах для создания отчёта

POLZA_PROVIDER = "polza"
POLZA_API_URL = "https://api.polza.ai/api/v1"

# Оставляем имя константы для внутренних импортов старого кода, но registry
# намеренно содержит только один разрешённый runtime-провайдер.
PROVIDERS: dict[str, dict[str, str]] = {
    POLZA_PROVIDER: {
        "token_env": "AI_TOKEN",
        "base_url": POLZA_API_URL,
    }
}


@dataclass(frozen=True, slots=True, init=False)
class AISettings:
    """Полная конфигурация chat- и embedding-runtime.

    ``token`` и ``embedding_token`` являются внутренними значениями. Они
    исключены из ``repr`` и из публичного представления, чтобы случайный
    snapshot или диагностическое сообщение не раскрыли credentials.
    """

    provider: str
    api_url: str
    token: str | None = field(default=None, repr=False)
    chat_model: str
    mini_model: str = "openai/gpt-6-luna"
    embedding_provider: str = POLZA_PROVIDER
    embedding_api_url: str = POLZA_API_URL
    embedding_token: str | None = field(default=None, repr=False)
    embedding_model: str = "text-embedding-3-large"

    def __init__(
        self,
        provider: str = POLZA_PROVIDER,
        api_url: str = POLZA_API_URL,
        token: str | None = None,
        chat_model: str = "openai/gpt-6-luna",
        mini_model: str = "openai/gpt-6-luna",
        embedding_provider: str = POLZA_PROVIDER,
        embedding_api_url: str = POLZA_API_URL,
        embedding_token: str | None = None,
        embedding_model: str = "text-embedding-3-large",
        *,
        model: str | None = None,
    ) -> None:
        """Создаёт настройки, принимая ``model`` как legacy alias.

        Alias нужен только для совместимости старых внутренних импортов;
        каноническое поле runtime и БД называется ``chat_model``.
        """
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "api_url", api_url)
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "chat_model", model if model is not None else chat_model)
        object.__setattr__(self, "mini_model", mini_model)
        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "embedding_api_url", embedding_api_url)
        object.__setattr__(self, "embedding_token", embedding_token)
        object.__setattr__(self, "embedding_model", embedding_model)

    @property
    def model(self) -> str:
        """Legacy alias для chat-модели."""
        return self.chat_model

    def __repr__(self) -> str:
        """Публичное представление без credentials."""
        values = self.public_dict()
        rendered = ", ".join(f"{key}={value!r}" for key, value in values.items())
        return f"AISettings({rendered})"

    def public_dict(self) -> dict[str, str | None]:
        """Возвращает настройки без credentials."""
        return {
            "provider": self.provider,
            "api_url": self.api_url,
            "chat_model": self.chat_model,
            "mini_model": self.mini_model,
            "embedding_provider": self.embedding_provider,
            "embedding_api_url": self.embedding_api_url,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_env(cls) -> "AISettings":
        """Переопределяет значения по умолчанию переменными окружения."""
        defaults = cls()
        settings = cls(
            provider=os.getenv("AI_PROVIDER") or defaults.provider,
            api_url=os.getenv("AI_API_URL") or defaults.api_url,
            token=os.getenv("AI_TOKEN") or None,
            chat_model=os.getenv("AI_MODEL") or defaults.chat_model,
            mini_model=os.getenv("AI_MODEL_MINI") or defaults.mini_model,
            embedding_provider=(os.getenv("AI_EMBEDDING_PROVIDER") or defaults.embedding_provider),
            embedding_api_url=(os.getenv("AI_EMBEDDING_API_URL") or defaults.embedding_api_url),
            embedding_token=os.getenv("AI_EMBEDDING_TOKEN") or None,
            embedding_model=os.getenv("AI_EMBEDDING_MODEL") or defaults.embedding_model,
        )
        return validate_ai_settings(settings, require_tokens=False)


def get_ai_settings() -> AISettings:
    """Возвращает AI-настройки с учётом текущего окружения."""
    return AISettings.from_env()


def validate_ai_settings(settings: AISettings, *, require_tokens: bool = True) -> AISettings:
    """Проверяет Polza-only конфигурацию без вывода секретов."""
    if settings.provider != POLZA_PROVIDER:
        raise ValueError("AI provider должен быть polza")
    if settings.embedding_provider != POLZA_PROVIDER:
        raise ValueError("Embedding provider должен быть polza")

    for field_name, value in (
        ("api_url", settings.api_url),
        ("embedding_api_url", settings.embedding_api_url),
    ):
        if not isinstance(value, str):
            raise TypeError(f"Некорректный {field_name}")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Некорректный {field_name}")

    for field_name, value in (
        ("chat_model", settings.chat_model),
        ("mini_model", settings.mini_model),
        ("embedding_model", settings.embedding_model),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Некорректное поле {field_name}")

    if require_tokens:
        if not isinstance(settings.token, str) or not settings.token.strip():
            raise ValueError("Отсутствует chat token")
        if not isinstance(settings.embedding_token, str) or not settings.embedding_token.strip():
            raise ValueError("Отсутствует embedding token")
    return settings


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Настройки схемы PostgreSQL."""

    schema: str = "discord"

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        """Читает схему из окружения, сохраняя значение по умолчанию."""
        defaults = cls()
        return cls(schema=os.getenv("DATABASE_SCHEMA", defaults.schema))


def get_database_settings() -> DatabaseSettings:
    """Возвращает настройки схемы PostgreSQL из окружения."""
    return DatabaseSettings.from_env()


@dataclass(frozen=True, slots=True)
class SchedulerSettings:
    """Таймзона и интервал проверки новых видео."""

    timezone: str
    youtube_check_interval_minutes: int


def get_scheduler_settings() -> SchedulerSettings:
    """Проверяет и возвращает настройки планировщика из общего конфига."""
    try:
        pytz.timezone(BOT_TIMEZONE)
    except pytz.UnknownTimeZoneError as exc:
        raise ValueError(f"Некорректная таймзона BOT_TIMEZONE: {BOT_TIMEZONE!r}") from exc

    interval = YOUTUBE_CHECK_INTERVAL_MINUTES
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("YOUTUBE_CHECK_INTERVAL_MINUTES должно быть целым числом больше 0")

    return SchedulerSettings(timezone=BOT_TIMEZONE, youtube_check_interval_minutes=interval)


def get_command_prefix() -> str:
    """Возвращает префикс команд из общего конфига, не допуская пустого значения."""
    if not isinstance(COMMAND_PREFIX, str) or not COMMAND_PREFIX.strip():
        raise ValueError("COMMAND_PREFIX не может быть пустым")
    return COMMAND_PREFIX
