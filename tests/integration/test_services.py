"""Сервисы с настоящим DB-слоем, но без сетевых Discord/YouTube/AI API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import feedparser
import pytest

from app.data import models
from app.services import daily_report, youtube_notifier
from tests.integration.conftest import Databases


async def test_youtube_add_toggle_and_per_guild_deduplication(
    databases: Databases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Общий feed уведомляет каждый guild один раз после добавления каналов."""
    channels = {1000: SimpleNamespace(send=AsyncMock()), 2000: SimpleNamespace(send=AsyncMock())}
    bot = SimpleNamespace(get_channel=channels.get)
    notifier = youtube_notifier.YouTubeNotifier(bot)
    feed = SimpleNamespace(
        status=200,
        entries=[feedparser.FeedParserDict(
            yt_videoid="v1", author="Author", title="Live integration", link="https://youtu.be/v1"
        )],
    )
    parse = MagicMock(return_value=feed)
    monkeypatch.setattr(youtube_notifier.feedparser, "parse", parse)
    assert await notifier.add_channel("UC1", 1000, "news", 100) is True
    assert await notifier.add_channel("UC1", 1000, "news", 100) is False
    assert await notifier.add_channel("UC1", 2000, "news", 200) is True
    await notifier.check_new_videos()
    await notifier.check_new_videos()
    for channel in channels.values():
        channel.send.assert_awaited_once()
        assert "https://youtu.be/v1" in channel.send.await_args.args[0]
        assert "Прямой эфир" in channel.send.await_args.args[0]
    assert parse.call_count == 4
    assert {
        (r.guild_id, r.video_id, r.is_live)
        for r in await databases.rows(models.YouTubeVideo)
    } == {(100, "v1", True), (200, "v1", True)}
    assert databases.engine.pool.checkedout() == 0



async def test_youtube_toggle_updates_database(databases: Databases) -> None:
    """Переключение канала в обе стороны сохраняется в БД."""
    notifier = youtube_notifier.YouTubeNotifier(SimpleNamespace(get_channel=lambda _: None))
    assert await notifier.add_channel("UC1", 1000, "news", 100) is True

    assert await notifier.toggle_channel("news", 100, False) is True
    rows = await databases.rows(models.YouTubeChannel)
    assert [(row.guild_id, row.is_active) for row in rows] == [(100, False)]
    assert await notifier.toggle_channel("news", 100, True) is True
    rows = await databases.rows(models.YouTubeChannel)
    assert [(row.guild_id, row.is_active) for row in rows] == [(100, True)]
    assert await notifier.toggle_channel("missing", 100, False) is None


async def test_youtube_toggle_uses_subscription_key(databases: Databases) -> None:
    """Разные ID/имена и дубликаты не мешают обновить только заданную подписку."""
    await databases.seed(
        models.YouTubeChannel(
            id=10, channel_id="UC1", discord_channel_id=1000, guild_id=100,
            name="news", is_active=True,
        ),
        models.YouTubeChannel(
            id=11, channel_id="UC1", discord_channel_id=1000, guild_id=100,
            name="alias", is_active=True,
        ),
        models.YouTubeChannel(
            id=30, channel_id="UC1", discord_channel_id=1000, guild_id=200,
            name="news", is_active=True,
        ),
        models.YouTubeChannel(
            id=40, channel_id="UC1", discord_channel_id=2000, guild_id=100,
            name="other destination", is_active=True,
        ),
        models.YouTubeChannel(
            id=50, channel_id="UC2", discord_channel_id=1000, guild_id=100,
            name="other channel", is_active=True,
        ),
    )
    notifier = youtube_notifier.YouTubeNotifier(SimpleNamespace())
    for active in (False, False, True):
        assert await notifier.toggle_channel("news", 100, active) is True
        rows = await databases.rows(models.YouTubeChannel)
        expected = {10: active, 11: active, 30: True, 40: True, 50: True}
        assert len(rows) == len(expected)
        assert {row.id: row.is_active for row in rows} == expected


async def test_youtube_toggle_rejects_ambiguous_name(
    databases: Databases, capsys: pytest.CaptureFixture[str]
) -> None:
    """Неоднозначное имя не выбирается произвольно."""
    await databases.seed(
        *[
            models.YouTubeChannel(
                channel_id=channel_id, discord_channel_id=1000, guild_id=100,
                name="news", is_active=True,
            )
            for channel_id in ("UC1", "UC2")
        ],
    )
    notifier = youtube_notifier.YouTubeNotifier(SimpleNamespace())
    assert await notifier.toggle_channel("news", 100, False) is None



async def test_youtube_discord_failure_does_not_mark_video_processed(
    databases: Databases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ошибка Discord до commit не помечает видео обработанным."""
    channel = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("Discord unavailable")))
    notifier = youtube_notifier.YouTubeNotifier(SimpleNamespace(get_channel=lambda _: channel))
    feed = SimpleNamespace(
        status=200,
        entries=[feedparser.FeedParserDict(
            yt_videoid="v1", author="Author", title="Video", link="https://youtu.be/v1"
        )],
    )
    monkeypatch.setattr(youtube_notifier.feedparser, "parse", MagicMock(return_value=feed))
    assert await notifier.add_channel("UC1", 1000, "news", 100)
    await notifier.check_new_videos()
    assert await databases.rows(models.YouTubeVideo) == []
    assert databases.engine.pool.checkedout() == 0
    channel.send.side_effect = None
    await notifier.check_new_videos()
    row, = await databases.rows(models.YouTubeVideo)
    assert row.video_id == "v1"
    assert row.is_live is False


@pytest.mark.parametrize("failure", [None, "ai", "discord"])
async def test_daily_report_reads_database_and_only_cleans_up_after_sending(
    databases: Databases, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Отчёт строится из БД; ошибки API сохраняют данные для повторной попытки."""
    channel = SimpleNamespace(guild=SimpleNamespace(id=10), send=AsyncMock())
    bot = SimpleNamespace(get_channel=lambda _: channel, report_msg_limit=100, report_time_limit=60)
    generator = daily_report.ReportGenerator(bot)
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Отчёт [ID:1001]"))]
        )
    )
    monkeypatch.setattr(
        daily_report, "get_client",
        lambda guild_id: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(daily_report, "get_mini_model", lambda guild_id: "test-model")
    await generator.add_message(100, "Привет", "alice", 1001)
    await generator.add_message(100, "Ответ", "bob", 1002)
    await generator.add_message(200, "Чужой канал", "charlie", 2001)
    assert all(state.timer is None for state in generator.channels.values())
    assert len(await databases.rows(models.ChannelMessage)) == 3
    generator.get_state(100).messages.clear()
    bot.report_msg_limit = 2
    if failure == "ai":
        create.side_effect = RuntimeError("AI unavailable")
    elif failure == "discord":
        channel.send.side_effect = RuntimeError("Discord unavailable")
    await generator.generate_and_send_report(100)
    create.assert_awaited_once()
    prompt = create.await_args.kwargs["messages"][1]["content"]
    assert "[ID:1001] alice: Привет" in prompt
    assert "[ID:1002] bob: Ответ" in prompt
    assert "Чужой канал" not in prompt
    if failure:
        assert 100 in generator.channels
        assert len(await databases.rows(models.ChannelMessage)) == 3
        if failure == "ai":
            channel.send.assert_not_awaited()
        create.side_effect = None
        channel.send.side_effect = None
        await generator.generate_and_send_report(100)
    assert 100 not in generator.channels
    assert 200 in generator.channels
    assert channel.send.await_args.args[0] == (
        "Отчёт [ссылка](https://discord.com/channels/10/100/1001)"
    )
    row, = await databases.rows(models.ChannelMessage)
    assert row.channel_id == 200
    assert databases.engine.pool.checkedout() == 0

