"""Тесты общих настроек приложения."""

import pytest

from app.core.config import DatabaseSettings, get_database_settings


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
