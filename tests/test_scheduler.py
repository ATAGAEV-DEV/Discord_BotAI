"""Тесты создания задания планировщика."""

from unittest.mock import MagicMock, patch

import pytest
import pytz

from app.core import config
from app.core.scheduler import start_scheduler


@pytest.mark.parametrize(("timezone", "interval"), [("Europe/Moscow", 5), ("Asia/Tokyo", 12)])
@patch("app.core.scheduler.AsyncIOScheduler")
def test_scheduler_uses_configured_timezone_and_interval(
    scheduler_class: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    timezone: str,
    interval: int,
) -> None:
    """Задание использует константы конфига, игнорируя окружение."""
    monkeypatch.setattr(config, "BOT_TIMEZONE", timezone)
    monkeypatch.setattr(config, "YOUTUBE_CHECK_INTERVAL_MINUTES", interval)
    monkeypatch.setenv("BOT_TIMEZONE", "Mars/Olympus")
    monkeypatch.setenv("YOUTUBE_CHECK_INTERVAL_MINUTES", "0")

    notifier = MagicMock()
    start_scheduler(MagicMock(), notifier)

    scheduler_class.assert_called_once_with(timezone=pytz.timezone(timezone))
    scheduler_class.return_value.add_job.assert_called_once_with(
        notifier.check_new_videos, "interval", minutes=interval, id="youtube_check"
    )
    scheduler_class.return_value.start.assert_called_once_with()
