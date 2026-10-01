"""Дополнительные изолированные тесты общего Cog."""

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest
from discord.ext import commands

from app.cogs import general
from app.core.bot import DisBot


@pytest.mark.asyncio
async def test_rank_value_error_sends_message_and_exits_typing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError завершает typing и отправляет текст ошибки вместо карточки ранга."""
    cog = general.General(MagicMock(spec=DisBot))
    ctx = MagicMock(spec=commands.Context)
    ctx.author = SimpleNamespace(id=12345)
    ctx.guild = SimpleNamespace(id=67890)
    ctx.send = AsyncMock()
    typing = ctx.typing.return_value
    typing.__aenter__ = AsyncMock(return_value=None)
    typing.__aexit__ = AsyncMock(return_value=False)
    error = ValueError("Некорректное число сообщений.")
    get_rank = AsyncMock(side_effect=error)
    get_description = MagicMock()
    create_embed = AsyncMock()
    monkeypatch.setattr(general, "get_rank", get_rank)
    monkeypatch.setattr(general, "get_rank_description", get_description)
    monkeypatch.setattr(general, "em", SimpleNamespace(create_rang_embed=create_embed))

    await cog.rank_command.callback(cog, ctx)

    get_rank.assert_called_once_with(12345, 67890)
    get_rank.assert_awaited_once_with(12345, 67890)
    ctx.typing.assert_called_once_with()
    typing.__aenter__.assert_awaited_once_with()
    typing.__aexit__.assert_awaited_once_with(ValueError, error, ANY)
    ctx.send.assert_called_once_with(str(error))
    ctx.send.assert_awaited_once_with(str(error))
    get_description.assert_not_called()
    create_embed.assert_not_called()


@pytest.mark.asyncio
async def test_setup_registers_general_once() -> None:
    """Загрузка расширения ровно один раз ожидает регистрацию General с исходным ботом."""
    bot = MagicMock(spec=DisBot)
    bot.add_cog = AsyncMock()

    await general.setup(bot)

    bot.add_cog.assert_awaited_once()
    cog = bot.add_cog.await_args.args[0]
    assert isinstance(cog, general.General)
    assert cog.bot is bot
    assert bot.mock_calls == [call.add_cog(cog)]
