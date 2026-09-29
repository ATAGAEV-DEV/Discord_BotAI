"""Тесты запуска Discord-бота."""

from unittest.mock import MagicMock, patch

import pytest

import main as bot_main
from app.core import config


@patch("main.DisBot")
def test_main_passes_configured_command_prefix(
    bot_class: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Запуск передаёт префикс команд из общего конфига в DisBot."""
    monkeypatch.setenv("DC_TOKEN", "test-token")
    monkeypatch.setattr(config, "COMMAND_PREFIX", "~")
    monkeypatch.setenv("COMMAND_PREFIX", "!")

    bot_main.main()

    assert bot_class.call_args.kwargs["command_prefix"] == "~"
    bot_class.return_value.run.assert_called_once_with("test-token")


@pytest.mark.parametrize(
    ("context_limit", "report_msg_limit", "report_time_limit"),
    [(100, 15, 60), (42, 4, 10)],
)
@patch("main.DisBot")
def test_main_passes_limits_from_config(
    bot_class: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    context_limit: int,
    report_msg_limit: int,
    report_time_limit: int,
) -> None:
    """Изменённые лимиты из общего конфига передаются боту при запуске."""
    monkeypatch.setenv("DC_TOKEN", "test-token")
    monkeypatch.setattr(config, "CONTEXT_LIMIT", context_limit)
    monkeypatch.setattr(config, "REPORT_MSG_LIMIT", report_msg_limit)
    monkeypatch.setattr(config, "REPORT_TIME_LIMIT", report_time_limit)

    bot_main.main()

    assert bot_class.call_args.kwargs["context_limit"] == context_limit
    assert bot_class.call_args.kwargs["report_msg_limit"] == report_msg_limit
    assert bot_class.call_args.kwargs["report_time_limit"] == report_time_limit
    bot_class.return_value.run.assert_called_once_with("test-token")
