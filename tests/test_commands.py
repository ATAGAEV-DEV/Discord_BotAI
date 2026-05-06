"""Unit-тесты для разбитых Cog-модулей."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands

from app.cogs.admin import Admin
from app.cogs.error_handler import ErrorHandler
from app.cogs.general import General
from app.cogs.youtube import YouTube
from app.core.bot import DisBot


@pytest.fixture
def mock_bot() -> MagicMock:
    """Фикстура для мока бота."""
    bot = MagicMock(spec=DisBot)
    bot.context_limit = 50
    bot.command_prefix = "!"
    bot.youtube_notifier = MagicMock()
    return bot


@pytest.fixture
def mock_ctx() -> AsyncMock:
    """Фикстура для мока контекста команды."""
    ctx = AsyncMock(spec=commands.Context)
    ctx.send = AsyncMock()
    ctx.author = MagicMock(spec=discord.Member)
    ctx.author.name = "test_user"
    ctx.author.display_name = "Test User"
    ctx.author.id = 12345
    ctx.author.avatar.url = "http://avatar.url"
    ctx.guild = MagicMock(spec=discord.Guild)
    ctx.guild.id = 67890
    ctx.guild.name = "Test Guild"
    ctx.message = MagicMock(spec=discord.Message)
    ctx.message.content = ""
    return ctx


@pytest.fixture
def general_cog(mock_bot: MagicMock) -> General:
    """Фикстура для General Cog."""
    return General(mock_bot)


@pytest.fixture
def admin_cog(mock_bot: MagicMock) -> Admin:
    """Фикстура для Admin Cog."""
    return Admin(mock_bot)


@pytest.fixture
def youtube_cog(mock_bot: MagicMock) -> YouTube:
    """Фикстура для YouTube Cog."""
    return YouTube(mock_bot)


@pytest.fixture
def error_cog(mock_bot: MagicMock) -> ErrorHandler:
    """Фикстура для ErrorHandler Cog."""
    return ErrorHandler(mock_bot)


# ── help_command ────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.cogs.general.em.create_help_embed")
async def test_help_command(
    mock_create_help: MagicMock, general_cog: General, mock_ctx: AsyncMock
) -> None:
    """Команда !help отправляет embed."""
    mock_embed = MagicMock()
    mock_create_help.return_value = mock_embed

    await general_cog.help_command.callback(general_cog, mock_ctx)

    mock_ctx.send.assert_called_once_with(embed=mock_embed)


# ── rank_command ────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.cogs.general.get_rank", new_callable=AsyncMock)
@patch("app.cogs.general.em.create_rang_embed", new_callable=AsyncMock)
async def test_rank_command_show_user(
    mock_create_embed: AsyncMock,
    mock_get_rank: AsyncMock,
    general_cog: General,
    mock_ctx: AsyncMock,
) -> None:
    """Команда !rank показывает ранг пользователя."""
    mock_get_rank.return_value = 50
    mock_embed = MagicMock()
    mock_file = MagicMock()
    mock_create_embed.return_value = (mock_embed, mock_file)

    await general_cog.rank_command.callback(general_cog, mock_ctx)

    mock_get_rank.assert_called_once_with(12345, 67890)
    mock_ctx.send.assert_called_once_with(embed=mock_embed, file=mock_file)


@pytest.mark.asyncio
@patch("app.cogs.general.em.create_rang_list_embed")
async def test_rank_command_list(
    mock_create_list: MagicMock, general_cog: General, mock_ctx: AsyncMock
) -> None:
    """Команда !rank list показывает список рангов."""
    mock_embed = MagicMock()
    mock_create_list.return_value = mock_embed

    await general_cog.rank_command.callback(general_cog, mock_ctx, arg="list")

    mock_ctx.send.assert_called_once_with(embed=mock_embed)


@pytest.mark.asyncio
@patch("app.cogs.general.get_rank", new_callable=AsyncMock)
async def test_rank_command_error(
    mock_get_rank: AsyncMock, general_cog: General, mock_ctx: AsyncMock
) -> None:
    """Обработка ошибки в !rank."""
    mock_get_rank.side_effect = Exception("DB Error")

    await general_cog.rank_command.callback(general_cog, mock_ctx)

    mock_ctx.send.assert_called_with("Произошла ошибка при получении статистики: DB Error")


# ── reset_command ───────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.core.handlers.clear_server_history")
async def test_reset_command(mock_clear: AsyncMock, admin_cog: Admin, mock_ctx: AsyncMock) -> None:
    """Команда !reset очищает историю."""
    mock_clear.return_value = "История очищена"
    await admin_cog.reset_command.callback(admin_cog, mock_ctx)

    mock_clear.assert_called_once_with(67890)
    mock_ctx.send.assert_called_with("История очищена")


# ── update_user_command ─────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.core.handlers.llama_manager.index_server_users", new_callable=AsyncMock)
async def test_update_user_command(
    mock_index: AsyncMock, admin_cog: Admin, mock_ctx: AsyncMock
) -> None:
    """Команда !update_user индексирует пользователей."""
    member = MagicMock()
    member.name = "user1"
    member.bot = False
    mock_ctx.guild.members = [member]

    await admin_cog.update_user_command.callback(admin_cog, mock_ctx)

    mock_index.assert_called_once_with(67890, ["user1"])
    assert "Список пользователей сервера обновлен" in mock_ctx.send.call_args[0][0]


# ── ai_provider_command ─────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.cogs.admin.get_available_providers")
@patch("app.cogs.admin.next_provider")
async def test_ai_provider_next(
    mock_next: MagicMock,
    mock_available: MagicMock,
    admin_cog: Admin,
    mock_ctx: AsyncMock,
) -> None:
    """!ai без аргументов переключает на следующего провайдера."""
    mock_available.return_value = ["prov1", "prov2"]
    mock_next.return_value = "prov2"

    await admin_cog.ai_provider_command.callback(admin_cog, mock_ctx, name=None)

    mock_next.assert_called_once()
    mock_ctx.send.assert_called()
    assert "prov2" in mock_ctx.send.call_args[0][0]


@pytest.mark.asyncio
@patch("app.cogs.admin.get_available_providers")
@patch("app.cogs.admin.set_active_provider")
async def test_ai_provider_set_valid(
    mock_set: MagicMock,
    mock_available: MagicMock,
    admin_cog: Admin,
    mock_ctx: AsyncMock,
) -> None:
    """!ai name устанавливает провайдера."""
    mock_available.return_value = ["prov1", "prov2"]

    await admin_cog.ai_provider_command.callback(admin_cog, mock_ctx, name="prov1")

    mock_set.assert_called_once_with("prov1")
    assert "переключён на **prov1**" in mock_ctx.send.call_args[0][0]


@pytest.mark.asyncio
@patch("app.cogs.admin.get_available_providers")
async def test_ai_provider_invalid(
    mock_available: MagicMock, admin_cog: Admin, mock_ctx: AsyncMock
) -> None:
    """!ai invalid сообщает об ошибке."""
    mock_available.return_value = ["prov1", "prov2"]

    await admin_cog.ai_provider_command.callback(admin_cog, mock_ctx, name="invalid")

    assert "не найден" in mock_ctx.send.call_args[0][0]


# ── youtube_commands ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_youtube_command_success(youtube_cog: YouTube, mock_ctx: AsyncMock) -> None:
    """!add_youtube добавляет канал."""
    youtube_cog.youtube_notifier.add_channel = AsyncMock(return_value=True)
    youtube_cog.bot.get_channel.return_value = MagicMock()

    await youtube_cog.add_youtube_command.callback(
        youtube_cog, mock_ctx, youtube_id="yt123", discord_channel_id=111, name="MyChannel"
    )

    youtube_cog.youtube_notifier.add_channel.assert_called_once()
    mock_ctx.send.assert_called_with("✅ Канал добавлен для отслеживания")


@pytest.mark.asyncio
async def test_youtube_toggle_command(youtube_cog: YouTube, mock_ctx: AsyncMock) -> None:
    """!youtube on/off переключает статус."""
    youtube_cog.youtube_notifier.toggle_channel = AsyncMock(return_value=True)

    await youtube_cog.youtube_toggle_command.callback(
        youtube_cog, mock_ctx, action="off", name="MyChannel"
    )

    youtube_cog.youtube_notifier.toggle_channel.assert_called_once_with("MyChannel", 67890, False)
    assert "отключено" in mock_ctx.send.call_args[0][0]


# ── on_command_error ────────────────────────────────────────────


@pytest.mark.asyncio
@patch("app.core.handlers.ai_generate", new_callable=AsyncMock)
async def test_on_command_not_found_generates_ai_response(
    mock_ai: AsyncMock,
    error_cog: ErrorHandler,
    mock_ctx: AsyncMock,
) -> None:
    """Неизвестная команда вызывает AI-генерацию."""
    error = commands.CommandNotFound()
    mock_ctx.message.content = "Как дела?"
    mock_ai.return_value = "Нормально"

    await error_cog.on_command_error(mock_ctx, error)

    mock_ai.assert_called_once()
    mock_ctx.send.assert_called()
    assert "Нормально" in mock_ctx.send.call_args[0][0]


@pytest.mark.asyncio
async def test_on_missing_permissions(error_cog: ErrorHandler, mock_ctx: AsyncMock) -> None:
    """Ошибка прав доступа."""
    error = commands.MissingPermissions(["administrator"])
    await error_cog.on_command_error(mock_ctx, error)
    assert "недостаточно прав" in mock_ctx.send.call_args[0][0]
