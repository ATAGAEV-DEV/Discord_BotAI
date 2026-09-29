"""Тесты уведомлений о новых видео."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from app.core import config
from app.services.youtube_notifier import YouTubeNotifier


@pytest.mark.asyncio
@patch("app.services.youtube_notifier.datetime")
@patch("app.services.youtube_notifier.asyncio.to_thread", new_callable=AsyncMock)
async def test_unavailable_feed_log_uses_configured_timezone(
    mock_to_thread: AsyncMock,
    mock_datetime: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Метка времени при ошибке загрузки YouTube использует общую таймзону."""
    monkeypatch.setattr(config, "BOT_TIMEZONE", "Asia/Tokyo")
    mock_to_thread.return_value = MagicMock(status=404, entries=[])
    channel = MagicMock(channel_id="UC123", name="Test")

    await YouTubeNotifier(MagicMock())._check_channel_videos(channel)

    mock_datetime.now.assert_called_once_with(tz=pytz.timezone("Asia/Tokyo"))
