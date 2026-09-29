"""Тесты общих настроек приложения."""

import pytest

from app.core import config
from app.core.config import (
    DatabaseSettings,
    SchedulerSettings,
    get_command_prefix,
    get_database_settings,
    get_scheduler_settings,
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
