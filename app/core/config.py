import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

AI_GENERATE_TIMEOUT: float = 100.0  # Секунды для генерации ответа и запроса к AI API
MAX_MESSAGE_LENGTH: int = 1000  # Максимальная длина входящего сообщения для отчетов
REPORT_IGNORE_PREFIX: str = "?"  # Сообщения с этим префиксом не входят в отчёты

PROVIDERS: dict[str, dict[str, str]] = {
    "proxyapi": {
        "token_env": "AI_TOKEN",
        "base_url": "https://api.proxyapi.ru/openai/v1",
    },
    "aitunnel": {
        "token_env": "AI_TOKEN1",
        "base_url": "https://api.aitunnel.ru/v1/",
    },
    "polza": {
        "token_env": "AI_TOKEN_POLZA",
        "base_url": "https://api.polza.ai/api/v1",
    },
    "vibecode": {
        "token_env": "AI_TOKEN_VIBECODE",
        "base_url": "https://api.vibecode-claude.online/v1",
    },
}


@dataclass(frozen=True, slots=True)
class AISettings:
    """Настройки провайдеров и моделей AI."""

    provider: str = "polza"
    model: str = "openai/gpt-6-luna"
    mini_model: str = "openai/gpt-6-luna"
    embedding_provider: str = "polza"
    embedding_model: str = "text-embedding-3-large"

    @classmethod
    def from_env(cls) -> "AISettings":
        """Переопределяет значения по умолчанию переменными окружения."""
        defaults = cls()
        return cls(
            provider=os.getenv("AI_PROVIDER", defaults.provider),
            model=os.getenv("AI_MODEL", defaults.model),
            mini_model=os.getenv("AI_MODEL_MINI", defaults.mini_model),
            embedding_provider=os.getenv("AI_EMBEDDING_PROVIDER", defaults.embedding_provider),
            embedding_model=os.getenv("AI_EMBEDDING_MODEL", defaults.embedding_model),
        )


def get_ai_settings() -> AISettings:
    """Возвращает AI-настройки с учётом текущего окружения."""
    return AISettings.from_env()


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
