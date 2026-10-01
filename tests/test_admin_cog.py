"""Дополнительные изолированные тесты административного Cog."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from discord.ext import commands

from app.cogs import admin
from app.core.bot import DisBot


@pytest.mark.asyncio
async def test_ai_provider_already_active_does_not_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Повторный выбор активного провайдера только отправляет уведомление."""
    bot = MagicMock(spec=DisBot)
    cog = admin.Admin(bot)
    ctx = MagicMock(spec=commands.Context)
    ctx.send = AsyncMock()
    get_available = MagicMock(return_value=["prov1", "prov2"])
    get_active = MagicMock(return_value="prov1")
    set_active = MagicMock()
    next_provider = MagicMock()
    monkeypatch.setattr(admin, "get_available_providers", get_available)
    monkeypatch.setattr(admin, "get_active_provider", get_active)
    monkeypatch.setattr(admin, "set_active_provider", set_active)
    monkeypatch.setattr(admin, "next_provider", next_provider)

    await cog.ai_provider_command.callback(cog, ctx, name="prov1")

    get_available.assert_called_once_with()
    get_active.assert_called_once_with()
    ctx.send.assert_called_once_with("ℹ️ Провайдер **prov1** уже активен.")
    ctx.send.assert_awaited_once_with("ℹ️ Провайдер **prov1** уже активен.")
    set_active.assert_not_called()
    next_provider.assert_not_called()
    assert bot.mock_calls == []


@pytest.mark.asyncio
async def test_setup_registers_admin_once() -> None:
    """Загрузка расширения ровно один раз ожидает регистрацию Admin с исходным ботом."""
    bot = MagicMock(spec=DisBot)
    bot.add_cog = AsyncMock()

    await admin.setup(bot)

    bot.add_cog.assert_awaited_once()
    cog = bot.add_cog.await_args.args[0]
    assert isinstance(cog, admin.Admin)
    assert cog.bot is bot
    assert bot.mock_calls == [call.add_cog(cog)]
