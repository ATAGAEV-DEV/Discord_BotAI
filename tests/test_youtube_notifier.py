"""Тесты уведомлений о новых видео."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import pytz

from app.core import config
from app.data.models import YouTubeChannel, YouTubeVideo
from app.services import youtube_notifier
from app.services.youtube_notifier import YouTubeNotifier


@pytest.fixture
def database_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Подменяет фабрику сессий контролируемым async-контекстом."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.context_entered = False
    session.context_exited = False

    @asynccontextmanager
    async def fake_async_session() -> AsyncIterator[MagicMock]:
        session.context_entered = True
        try:
            yield session
        finally:
            session.context_exited = True

    monkeypatch.setattr(youtube_notifier, "async_session", fake_async_session)
    return session


def _rows_result(*rows: object) -> MagicMock:
    """Создаёт результат SQLAlchemy со списком строк."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(rows)
    return result


def _scalar_result(value: object) -> MagicMock:
    """Создаёт результат SQLAlchemy с одной записью или значением None."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _feed_entry(
    title: str,
    video_id: str = "video-123",
    author: str = "Author",
    link: str = "https://youtu.be/video-123",
) -> MagicMock:
    """Создаёт запись YouTube feed с API, используемым сервисом."""
    entry = MagicMock()
    values = {"yt_videoid": video_id, "author": author}
    entry.get.side_effect = values.get
    entry.title = title
    entry.link = link
    return entry


def _channel(
    channel_id: str = "UC123",
    name: str = "Test channel",
    guild_id: int = 100,
    discord_channel_id: int = 200,
) -> SimpleNamespace:
    """Создаёт описание отслеживаемого YouTube-канала."""
    return SimpleNamespace(
        channel_id=channel_id,
        name=name,
        guild_id=guild_id,
        discord_channel_id=discord_channel_id,
    )


@pytest.mark.asyncio
async def test_check_new_videos_with_no_active_channels(database_session: MagicMock) -> None:
    """Проверка без активных каналов не вызывает проверку отдельных каналов."""
    database_session.execute.return_value = _rows_result()
    notifier = YouTubeNotifier(MagicMock())
    notifier._check_channel_videos = AsyncMock()

    await notifier.check_new_videos()

    database_session.execute.assert_awaited_once()
    notifier._check_channel_videos.assert_not_awaited()
    assert database_session.context_entered is True
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_check_new_videos_checks_each_active_channel(
    database_session: MagicMock,
) -> None:
    """Все активные каналы передаются во внутреннюю проверку по очереди."""
    first_channel = _channel(channel_id="UC1")
    second_channel = _channel(channel_id="UC2")
    database_session.execute.return_value = _rows_result(first_channel, second_channel)
    notifier = YouTubeNotifier(MagicMock())
    notifier._check_channel_videos = AsyncMock()

    await notifier.check_new_videos()

    assert notifier._check_channel_videos.await_args_list == [
        call(first_channel),
        call(second_channel),
    ]


@pytest.mark.asyncio
async def test_check_new_videos_logs_unexpected_error(
    database_session: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Непредвиденная ошибка общей проверки выводится и не пробрасывается наружу."""
    database_session.execute.side_effect = ValueError("database unavailable")

    await YouTubeNotifier(MagicMock()).check_new_videos()

    assert "Ошибка при проверке YouTube видео: database unavailable" in capsys.readouterr().out
    assert database_session.context_exited is True


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
    channel = _channel()

    await YouTubeNotifier(MagicMock())._check_channel_videos(channel)

    mock_to_thread.assert_awaited_once_with(
        youtube_notifier.feedparser.parse,
        "https://www.youtube.com/feeds/videos.xml?channel_id=UC123",
    )
    mock_datetime.now.assert_called_once_with(tz=pytz.timezone("Asia/Tokyo"))


@pytest.mark.asyncio
async def test_empty_successful_feed_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """Успешный HTTP-ответ без записей считается недоступным feed."""
    with patch.object(youtube_notifier.asyncio, "to_thread", new_callable=AsyncMock) as mock:
        mock.return_value = SimpleNamespace(status=200, entries=[])

        await YouTubeNotifier(MagicMock())._check_channel_videos(_channel())

    assert "Неверный или недоступный канал: Test channel" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_existing_video_is_not_added_but_transaction_is_committed(
    database_session: MagicMock,
) -> None:
    """Уже сохранённое видео не дублируется, но проверка фиксирует транзакцию."""
    database_session.execute.return_value = _scalar_result(SimpleNamespace(video_id="video-123"))
    entry = _feed_entry("Existing video")

    with patch.object(youtube_notifier.asyncio, "to_thread", new_callable=AsyncMock) as mock:
        mock.return_value = SimpleNamespace(status=200, entries=[entry])

        bot = MagicMock()
        await YouTubeNotifier(bot)._check_channel_videos(_channel())

    database_session.add.assert_not_called()
    database_session.commit.assert_awaited_once_with()
    bot.get_channel.assert_not_called()
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_new_regular_video_is_saved_and_sent_to_discord(
    database_session: MagicMock,
) -> None:
    """Новое обычное видео сохраняется и отправляется в найденный Discord-канал."""
    database_session.execute.return_value = _scalar_result(None)
    entry = _feed_entry("New video")
    discord_channel = MagicMock()
    discord_channel.send = AsyncMock()
    bot = MagicMock()
    bot.get_channel.return_value = discord_channel
    channel = _channel()

    with patch.object(youtube_notifier.asyncio, "to_thread", new_callable=AsyncMock) as mock:
        mock.return_value = SimpleNamespace(status=200, entries=[entry])

        await YouTubeNotifier(bot)._check_channel_videos(channel)

    new_video = database_session.add.call_args.args[0]
    assert isinstance(new_video, YouTubeVideo)
    assert new_video.video_id == "video-123"
    assert new_video.guild_id == 100
    assert new_video.channel_id == "UC123"
    assert new_video.title == "New video"
    assert new_video.published_at.tzinfo is None
    assert new_video.is_live is False
    bot.get_channel.assert_called_once_with(200)
    discord_channel.send.assert_awaited_once_with(
        "🎥 **Новое видео на канале [Author](https://www.youtube.com/channel/UC123)!**\n"
        "https://youtu.be/video-123"
    )
    database_session.commit.assert_awaited_once_with()
    mock.assert_awaited_once_with(
        youtube_notifier.feedparser.parse,
        "https://www.youtube.com/feeds/videos.xml?channel_id=UC123",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("title", ["Live stream", "Новая прямая трансляция"])
async def test_new_live_video_uses_live_notification(
    title: str,
    database_session: MagicMock,
) -> None:
    """Оба поддерживаемых маркера прямого эфира дают live-уведомление."""
    database_session.execute.return_value = _scalar_result(None)
    entry = _feed_entry(title)
    discord_channel = MagicMock()
    discord_channel.send = AsyncMock()
    bot = MagicMock()
    bot.get_channel.return_value = discord_channel

    with patch.object(youtube_notifier.asyncio, "to_thread", new_callable=AsyncMock) as mock:
        mock.return_value = SimpleNamespace(status=200, entries=[entry])

        await YouTubeNotifier(bot)._check_channel_videos(_channel())

    new_video = database_session.add.call_args.args[0]
    assert new_video.is_live is True
    discord_channel.send.assert_awaited_once_with(
        "🔴 **Прямой эфир на канале [Author](https://www.youtube.com/channel/UC123)!**\n"
        "https://youtu.be/video-123"
    )


@pytest.mark.asyncio
async def test_new_video_without_discord_channel_is_still_saved(
    database_session: MagicMock,
) -> None:
    """Отсутствие Discord-канала не мешает сохранить новое видео."""
    database_session.execute.return_value = _scalar_result(None)
    bot = MagicMock()
    bot.get_channel.return_value = None
    entry = _feed_entry("New video")

    with patch.object(youtube_notifier.asyncio, "to_thread", new_callable=AsyncMock) as mock:
        mock.return_value = SimpleNamespace(status=200, entries=[entry])

        await YouTubeNotifier(bot)._check_channel_videos(_channel())

    assert isinstance(database_session.add.call_args.args[0], YouTubeVideo)
    database_session.commit.assert_awaited_once_with()
    bot.get_channel.assert_called_once_with(200)


@pytest.mark.asyncio
async def test_check_channel_videos_propagates_commit_error(
    database_session: MagicMock,
) -> None:
    """Ошибка фиксации видео передаётся общей проверке каналов."""
    database_session.execute.return_value = _scalar_result(None)
    database_session.commit.side_effect = RuntimeError("commit failed")
    entry = _feed_entry("New video")
    bot = MagicMock()
    bot.get_channel.return_value = None

    with patch.object(youtube_notifier.asyncio, "to_thread", new_callable=AsyncMock) as mock:
        mock.return_value = SimpleNamespace(status=200, entries=[entry])

        with pytest.raises(RuntimeError, match="commit failed"):
            await YouTubeNotifier(bot)._check_channel_videos(_channel())

    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_toggle_channel_updates_existing_channel(
    database_session: MagicMock,
) -> None:
    """Переключение найденного канала меняет статус и фиксирует транзакцию."""
    channel = SimpleNamespace(is_active=True)
    database_session.execute.return_value = _scalar_result(channel)

    result = await YouTubeNotifier(MagicMock()).toggle_channel("Test", 100, False)

    assert result is True
    assert channel.is_active is False
    database_session.commit.assert_awaited_once_with()
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_toggle_channel_returns_none_for_missing_channel(
    database_session: MagicMock,
) -> None:
    """Переключение отсутствующего канала возвращает None без commit."""
    database_session.execute.return_value = _scalar_result(None)

    result = await YouTubeNotifier(MagicMock()).toggle_channel("Missing", 100, True)

    assert result is None
    database_session.commit.assert_not_awaited()
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_toggle_channel_logs_commit_error(
    database_session: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка commit при переключении возвращает False и логируется."""
    database_session.execute.return_value = _scalar_result(SimpleNamespace(is_active=True))
    database_session.commit.side_effect = RuntimeError("commit failed")

    result = await YouTubeNotifier(MagicMock()).toggle_channel("Test", 100, False)

    assert result is False
    assert "Ошибка при переключении YouTube канала: commit failed" in capsys.readouterr().out
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_add_channel_creates_and_commits_new_channel(
    database_session: MagicMock,
) -> None:
    """Новый YouTube-канал создаётся с переданными параметрами."""
    database_session.execute.return_value = _scalar_result(None)

    result = await YouTubeNotifier(MagicMock()).add_channel("UC123", 200, "Test", 100)

    assert result is True
    new_channel = database_session.add.call_args.args[0]
    assert isinstance(new_channel, YouTubeChannel)
    assert new_channel.channel_id == "UC123"
    assert new_channel.discord_channel_id == 200
    assert new_channel.name == "Test"
    assert new_channel.guild_id == 100
    database_session.commit.assert_awaited_once_with()
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_add_channel_rejects_duplicate(
    database_session: MagicMock,
) -> None:
    """Дубликат канала отклоняется без изменения базы данных."""
    database_session.execute.return_value = _scalar_result(SimpleNamespace(channel_id="UC123"))

    result = await YouTubeNotifier(MagicMock()).add_channel("UC123", 200, "Test", 100)

    assert result is False
    database_session.add.assert_not_called()
    database_session.commit.assert_not_awaited()
    assert database_session.context_exited is True


@pytest.mark.asyncio
async def test_add_channel_logs_commit_error(
    database_session: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка commit при добавлении возвращает False и логируется."""
    database_session.execute.return_value = _scalar_result(None)
    database_session.commit.side_effect = RuntimeError("commit failed")

    result = await YouTubeNotifier(MagicMock()).add_channel("UC123", 200, "Test", 100)

    assert result is False
    assert "Ошибка при добавлении YouTube канала: commit failed" in capsys.readouterr().out
    assert database_session.context_exited is True
