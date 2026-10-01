"""Изолированные тесты команд YouTube и регистрации Cog."""

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from app.cogs.youtube import YouTube, setup
from app.core.bot import DisBot
from app.services.youtube_notifier import YouTubeNotifier


@pytest.fixture
def mock_bot() -> MagicMock:
    """Создаёт бот с подменённым notifier без Discord, БД и сетевых запросов."""
    bot = MagicMock(spec=DisBot)
    bot.command_prefix = "!"
    bot.youtube_notifier = MagicMock(spec=YouTubeNotifier)
    bot.youtube_notifier.add_channel = AsyncMock()
    bot.youtube_notifier.toggle_channel = AsyncMock()
    bot.get_channel.return_value = MagicMock(spec=discord.TextChannel)
    bot.add_cog = AsyncMock()
    return bot


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Создаёт контекст текущего сервера с асинхронной отправкой ответа."""
    ctx = MagicMock(spec=commands.Context)
    ctx.guild = MagicMock(spec=discord.Guild)
    ctx.guild.id = 67890
    ctx.send = AsyncMock()
    return ctx


@pytest.fixture
def youtube_cog(mock_bot: MagicMock) -> YouTube:
    """Создаёт реальный Cog с полностью подменёнными внешними зависимостями."""
    return YouTube(mock_bot)


def test_constructor_reuses_bot_notifier(mock_bot: MagicMock) -> None:
    """Cog сохраняет ссылки на бот и существующий notifier без вызова сервисов."""
    cog = YouTube(mock_bot)

    assert cog.bot is mock_bot
    assert cog.youtube_notifier is mock_bot.youtube_notifier
    assert mock_bot.youtube_notifier.mock_calls == []
    mock_bot.get_channel.assert_not_called()
    mock_bot.add_cog.assert_not_called()


@pytest.mark.asyncio
async def test_setup_registers_youtube_cog(mock_bot: MagicMock) -> None:
    """Загрузка расширения создаёт Cog и ожидает его регистрацию ровно один раз."""
    await setup(mock_bot)

    mock_bot.add_cog.assert_awaited_once()
    cog = mock_bot.add_cog.await_args.args[0]
    assert isinstance(cog, YouTube)
    assert cog.bot is mock_bot
    assert cog.youtube_notifier is mock_bot.youtube_notifier
    mock_bot.add_cog.assert_awaited_once_with(cog)
    assert mock_bot.youtube_notifier.mock_calls == []


@pytest.mark.asyncio
async def test_add_youtube_missing_discord_channel_skips_notifier(
    youtube_cog: YouTube, mock_bot: MagicMock, mock_ctx: MagicMock
) -> None:
    """Если Discord-канал не найден, команда отвечает ошибкой и не вызывает notifier."""
    mock_bot.get_channel.return_value = None

    await youtube_cog.add_youtube_command.callback(
        youtube_cog,
        mock_ctx,
        youtube_id="UC_test_channel",
        discord_channel_id=111,
        name="My Channel",
    )

    mock_bot.get_channel.assert_called_once_with(111)
    mock_bot.youtube_notifier.add_channel.assert_not_called()
    mock_bot.youtube_notifier.toggle_channel.assert_not_called()
    mock_ctx.send.assert_awaited_once_with("❌ Канал не найден!")


@pytest.mark.asyncio
@pytest.mark.parametrize("guild_id", [67890, 98765], ids=["first-guild", "second-guild"])
@pytest.mark.parametrize(
    ("result", "expected_message"),
    [
        (True, "✅ Канал добавлен для отслеживания"),
        (False, "❌ Ошибка при добавлении канала"),
        (None, "❌ Ошибка при добавлении канала"),
    ],
    ids=["success", "failure", "none"],
)
async def test_add_youtube_passes_arguments_and_reports_result(
    youtube_cog: YouTube,
    mock_bot: MagicMock,
    mock_ctx: MagicMock,
    guild_id: int,
    result: bool | None,
    expected_message: str,
) -> None:
    """Добавление передаёт все аргументы и ID текущего сервера, затем сообщает результат."""
    mock_ctx.guild.id = guild_id
    mock_bot.youtube_notifier.add_channel.return_value = result

    await youtube_cog.add_youtube_command.callback(
        youtube_cog,
        mock_ctx,
        youtube_id="UC_test_channel",
        discord_channel_id=111,
        name="My Channel",
    )

    mock_bot.get_channel.assert_called_once_with(111)
    mock_bot.youtube_notifier.add_channel.assert_awaited_once_with(
        "UC_test_channel", 111, "My Channel", guild_id
    )
    mock_bot.youtube_notifier.toggle_channel.assert_not_called()
    mock_ctx.send.assert_awaited_once_with(expected_message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "active", "expected_message"),
    [
        ("on", True, "✅ Отслеживание канала **My Channel** включено."),
        ("ON", True, "✅ Отслеживание канала **My Channel** включено."),
        ("oN", True, "✅ Отслеживание канала **My Channel** включено."),
        ("off", False, "⏸️ Отслеживание канала **My Channel** отключено."),
        ("OFF", False, "⏸️ Отслеживание канала **My Channel** отключено."),
        ("oFf", False, "⏸️ Отслеживание канала **My Channel** отключено."),
    ],
    ids=["on", "on-upper", "on-mixed", "off", "off-upper", "off-mixed"],
)
async def test_youtube_toggle_normalizes_action_and_reports_success(
    youtube_cog: YouTube,
    mock_bot: MagicMock,
    mock_ctx: MagicMock,
    action: str,
    active: bool,
    expected_message: str,
) -> None:
    """Регистр действия не влияет на статус; ответ содержит правильные текст и эмодзи."""
    mock_bot.youtube_notifier.toggle_channel.return_value = True

    await youtube_cog.youtube_toggle_command.callback(
        youtube_cog, mock_ctx, action=action, name="My Channel"
    )

    mock_bot.youtube_notifier.toggle_channel.assert_awaited_once_with("My Channel", 67890, active)
    mock_bot.youtube_notifier.add_channel.assert_not_called()
    mock_bot.get_channel.assert_not_called()
    mock_ctx.send.assert_awaited_once_with(expected_message)


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "active"), [("on", True), ("off", False)])
@pytest.mark.parametrize(
    ("result", "expected_message"),
    [
        (None, "❌ Канал **My Channel** не найден на этом сервере."),
        (False, "❌ Ошибка при изменении статуса канала."),
    ],
    ids=["not-found", "failure"],
)
async def test_youtube_toggle_distinguishes_missing_channel_and_failure(
    youtube_cog: YouTube,
    mock_bot: MagicMock,
    mock_ctx: MagicMock,
    action: str,
    active: bool,
    result: bool | None,
    expected_message: str,
) -> None:
    """Отсутствие канала отличается от ошибки изменения статуса для обоих действий."""
    mock_ctx.guild.id = 98765
    mock_bot.youtube_notifier.toggle_channel.return_value = result

    await youtube_cog.youtube_toggle_command.callback(
        youtube_cog, mock_ctx, action=action, name="My Channel"
    )

    mock_bot.youtube_notifier.toggle_channel.assert_awaited_once_with("My Channel", 98765, active)
    mock_bot.youtube_notifier.add_channel.assert_not_called()
    mock_bot.get_channel.assert_not_called()
    mock_ctx.send.assert_awaited_once_with(expected_message)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["!", "~"])
@pytest.mark.parametrize("action", ["enable", "", " on "], ids=["unknown", "empty", "whitespace"])
async def test_youtube_invalid_action_uses_prefix_without_calling_notifier(
    youtube_cog: YouTube,
    mock_bot: MagicMock,
    mock_ctx: MagicMock,
    prefix: str,
    action: str,
) -> None:
    """Неверное действие выводит подсказку с текущим префиксом без вызова notifier."""
    mock_bot.command_prefix = prefix

    await youtube_cog.youtube_toggle_command.callback(
        youtube_cog, mock_ctx, action=action, name="My Channel"
    )

    mock_bot.youtube_notifier.toggle_channel.assert_not_called()
    mock_bot.youtube_notifier.add_channel.assert_not_called()
    mock_bot.get_channel.assert_not_called()
    mock_ctx.send.assert_awaited_once_with(
        f"❌ Используйте: `{prefix}youtube on/off название_канала`"
    )
