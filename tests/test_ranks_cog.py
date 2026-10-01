"""Поведенческие тесты начисления опыта и уведомлений о повышении ранга."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from discord.ext import commands

from app.cogs import ranks


@pytest.fixture
def rank_env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Подменяет БД, генерацию карточек и Discord, сохраняя реальный выбор ранга."""
    bot = MagicMock(spec=commands.Bot)
    bot.command_prefix = "!"
    bot.add_cog = AsyncMock()
    message = SimpleNamespace(
        content="Обычное сообщение",
        author=SimpleNamespace(
            id=123,
            name="account_name",
            display_name="Имя на сервере",
            mention="<@123>",
            bot=False,
            avatar=SimpleNamespace(url="https://example.test/avatar.png"),
            default_avatar=SimpleNamespace(url="https://example.test/default.png"),
        ),
        guild=SimpleNamespace(id=456),
        channel=SimpleNamespace(send=AsyncMock()),
    )
    embed, file = object(), object()
    update = AsyncMock(return_value={"rank_up": False, "message_count": 49})
    create_embed = AsyncMock(return_value=(embed, file))
    monkeypatch.setattr(ranks, "update_message_count", update)
    monkeypatch.setattr(ranks, "em", SimpleNamespace(create_rang_embed=create_embed))
    return SimpleNamespace(
        bot=bot,
        cog=ranks.Ranks(bot),
        message=message,
        update=update,
        create_embed=create_embed,
        embed=embed,
        file=file,
    )


def test_constructor_keeps_bot_without_side_effects(rank_env: SimpleNamespace) -> None:
    """Создание Cog не начисляет опыт и не регистрирует себя самостоятельно."""
    assert rank_env.cog.bot is rank_env.bot
    rank_env.bot.add_cog.assert_not_called()
    rank_env.update.assert_not_called()
    rank_env.create_embed.assert_not_called()
    rank_env.message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_setup_registers_ranks_once(rank_env: SimpleNamespace) -> None:
    """Setup ожидает регистрацию ровно одного Cog с исходным ботом."""
    await ranks.setup(rank_env.bot)

    rank_env.bot.add_cog.assert_awaited_once()
    cog = rank_env.bot.add_cog.await_args.args[0]
    assert isinstance(cog, ranks.Ranks)
    assert cog.bot is rank_env.bot
    assert rank_env.bot.mock_calls == [call.add_cog(cog)]


@pytest.mark.asyncio
async def test_setup_propagates_registration_error(rank_env: SimpleNamespace) -> None:
    """Ошибка регистрации не скрывается и не вызывает повторную попытку."""
    error = RuntimeError("registration failed")
    rank_env.bot.add_cog.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await ranks.setup(rank_env.bot)

    assert exc_info.value is error
    rank_env.bot.add_cog.assert_awaited_once()
    rank_env.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_bot", "prefix", "content", "guild_id"),
    [
        (True, "!", "Обычное сообщение", 456),
        (False, "!", "!help", 456),
        (False, "~", "~help", 456),
        (False, "!", "Личное сообщение", None),
    ],
    ids=["bot", "command", "custom-prefix", "direct-message"],
)
async def test_ignored_messages_have_no_side_effects(
    rank_env: SimpleNamespace,
    is_bot: bool,
    prefix: str,
    content: str,
    guild_id: int | None,
) -> None:
    """Боты, команды с актуальным префиксом и личные сообщения не дают опыта."""
    rank_env.bot.command_prefix = prefix
    rank_env.message.author.bot = is_bot
    rank_env.message.content = content
    rank_env.message.guild = None if guild_id is None else SimpleNamespace(id=guild_id)

    await rank_env.cog.on_message(rank_env.message)

    rank_env.update.assert_not_called()
    rank_env.create_embed.assert_not_called()
    rank_env.message.channel.send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("guild_id", [456, 789])
@pytest.mark.parametrize("content", ["Обычное сообщение", "Используй !help для справки"])
async def test_message_without_rank_up_only_updates_experience(
    rank_env: SimpleNamespace, guild_id: int, content: str
) -> None:
    """Учёт использует ID сервера и имя аккаунта, но не имя отображения."""
    rank_env.message.guild.id = guild_id
    rank_env.message.content = content

    await rank_env.cog.on_message(rank_env.message)

    rank_env.update.assert_called_once_with(123, "account_name", guild_id)
    rank_env.update.assert_awaited_once_with(123, "account_name", guild_id)
    rank_env.create_embed.assert_not_called()
    rank_env.message.channel.send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("guild_id", [456, 789])
@pytest.mark.parametrize(
    ("message_count", "description"), [(50, "Радужный бич"), (100, "Бич")]
)
@pytest.mark.parametrize(
    ("has_avatar", "avatar_url"),
    [
        (True, "https://example.test/avatar.png"),
        (False, "https://example.test/default.png"),
    ],
    ids=["custom-avatar", "default-avatar"],
)
async def test_rank_up_sends_card_with_exact_user_and_rank_data(
    rank_env: SimpleNamespace,
    guild_id: int,
    message_count: int,
    description: str,
    has_avatar: bool,
    avatar_url: str,
) -> None:
    """Реальный выбор ранга, оба аватара и результат генератора доходят до отправки."""
    rank_env.message.guild.id = guild_id
    rank_env.update.return_value = {"rank_up": True, "message_count": message_count}
    if not has_avatar:
        rank_env.message.author.avatar = None
    operations = MagicMock()
    operations.attach_mock(rank_env.update, "update")
    operations.attach_mock(rank_env.create_embed, "create_embed")
    operations.attach_mock(rank_env.message.channel.send, "send")

    await rank_env.cog.on_message(rank_env.message)

    rank_env.update.assert_awaited_once_with(123, "account_name", guild_id)
    rank_env.create_embed.assert_awaited_once_with(
        "Имя на сервере", message_count, description, avatar_url, guild_id, 123
    )
    rank_env.message.channel.send.assert_awaited_once_with(
        "🎉 **<@123>** повысил свой ранг!", embed=rank_env.embed, file=rank_env.file
    )
    assert operations.mock_calls == [
        call.update(123, "account_name", guild_id),
        call.create_embed(
            "Имя на сервере", message_count, description, avatar_url, guild_id, 123
        ),
        call.send("🎉 **<@123>** повысил свой ранг!", embed=rank_env.embed, file=rank_env.file),
    ]


@pytest.mark.asyncio
async def test_database_error_stops_processing(rank_env: SimpleNamespace) -> None:
    """Ошибка БД не приводит к генерации карточки или ложному поздравлению."""
    error = RuntimeError("database failed")
    rank_env.update.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await rank_env.cog.on_message(rank_env.message)

    assert exc_info.value is error
    rank_env.update.assert_awaited_once_with(123, "account_name", 456)
    rank_env.create_embed.assert_not_called()
    rank_env.message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_card_error_does_not_send_congratulations(rank_env: SimpleNamespace) -> None:
    """Сбой генератора не скрывается; начисление опыта не повторяется."""
    rank_env.update.return_value = {"rank_up": True, "message_count": 50}
    error = RuntimeError("card failed")
    rank_env.create_embed.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await rank_env.cog.on_message(rank_env.message)

    assert exc_info.value is error
    rank_env.update.assert_awaited_once_with(123, "account_name", 456)
    rank_env.create_embed.assert_awaited_once()
    rank_env.message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_error_does_not_repeat_experience_or_card(rank_env: SimpleNamespace) -> None:
    """Сбой отправки распространяется без повторных побочных действий."""
    rank_env.update.return_value = {"rank_up": True, "message_count": 50}
    error = RuntimeError("send failed")
    rank_env.message.channel.send.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await rank_env.cog.on_message(rank_env.message)

    assert exc_info.value is error
    rank_env.update.assert_awaited_once_with(123, "account_name", 456)
    rank_env.create_embed.assert_awaited_once()
    rank_env.message.channel.send.assert_awaited_once_with(
        "🎉 **<@123>** повысил свой ранг!", embed=rank_env.embed, file=rank_env.file
    )
