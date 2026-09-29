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
