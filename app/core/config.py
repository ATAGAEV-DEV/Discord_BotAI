import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

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
