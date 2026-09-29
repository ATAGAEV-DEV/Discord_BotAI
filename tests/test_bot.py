"""Тесты обработки входящих сообщений ботом."""

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from app.core.bot import DisBot
from app.core.config import MAX_MESSAGE_LENGTH, REPORT_IGNORE_PREFIX


@pytest.fixture
def mock_bot() -> MagicMock:
    """Бот без подключения к Discord и внешним сервисам."""
    bot = MagicMock(spec=DisBot)
    bot.command_prefix = "!"
    bot.report_generator = MagicMock()
    bot.report_generator.add_message = AsyncMock()
    bot.process_commands = AsyncMock()
    return bot


def _message(content: str) -> MagicMock:
    """Создаёт сообщение с необходимыми для on_message полями."""
    message = MagicMock(spec=discord.Message)
    message.author.bot = False
    message.author.display_name = "Test User"
    message.content = content
    message.channel.id = 123
    message.channel.send = AsyncMock()
    message.id = 456
    return message


@pytest.mark.asyncio
async def test_message_at_maximum_length_is_accepted(mock_bot: MagicMock) -> None:
    """Сообщение ровно предельной длины включается в отчёт."""
    content = "a" * MAX_MESSAGE_LENGTH
    message = _message(content)

    await DisBot.on_message(mock_bot, message)

    mock_bot.report_generator.add_message.assert_awaited_once_with(123, content, "Test User", 456)
    message.channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_overlong_command_reports_configured_limit(mock_bot: MagicMock) -> None:
    """Команда сверх лимита отклоняется с актуальным значением ограничения."""
    message = _message("!" + "a" * MAX_MESSAGE_LENGTH)

    await DisBot.on_message(mock_bot, message)

    message.channel.send.assert_awaited_once_with(
        f"Сообщение слишком длинное: {len(message.content)} символов! "
        f"Максимальная длина - {MAX_MESSAGE_LENGTH} символов."
    )
    mock_bot.process_commands.assert_not_awaited()
    mock_bot.report_generator.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_overlong_noncommand_is_skipped(mock_bot: MagicMock) -> None:
    """Длинное обычное сообщение не попадает в отчёт и не выдаёт предупреждение."""
    message = _message("a" * (MAX_MESSAGE_LENGTH + 1))

    await DisBot.on_message(mock_bot, message)

    mock_bot.report_generator.add_message.assert_not_awaited()
    message.channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_short_command_still_runs(mock_bot: MagicMock) -> None:
    """Команда допустимой длины обрабатывается без добавления в отчёт."""
    message = _message("!help")

    await DisBot.on_message(mock_bot, message)

    mock_bot.process_commands.assert_awaited_once_with(message)
    mock_bot.report_generator.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_report_ignore_prefix_skips_message(mock_bot: MagicMock) -> None:
    """Сообщение с настроенным префиксом не включается в отчёт."""
    message = _message(f"{REPORT_IGNORE_PREFIX}скрыть")

    await DisBot.on_message(mock_bot, message)

    mock_bot.report_generator.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_report_ignore_prefix(
    mock_bot: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """При другом префиксе обычное сообщение включается, а помеченное — нет."""
    monkeypatch.setattr("app.core.bot.REPORT_IGNORE_PREFIX", "~")

    await DisBot.on_message(mock_bot, _message("?обычное сообщение"))
    mock_bot.report_generator.add_message.assert_awaited_once()

    await DisBot.on_message(mock_bot, _message("~пропустить"))
    mock_bot.report_generator.add_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_report_ignore_prefix_disables_filter(
    mock_bot: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Пустой префикс не исключает все сообщения из отчётов."""
    monkeypatch.setattr("app.core.bot.REPORT_IGNORE_PREFIX", "")
    message = _message("обычное сообщение")

    await DisBot.on_message(mock_bot, message)

    mock_bot.report_generator.add_message.assert_awaited_once_with(
        123, message.content, "Test User", 456
    )
