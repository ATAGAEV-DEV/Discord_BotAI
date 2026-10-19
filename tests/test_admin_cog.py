"""Дополнительные изолированные тесты административного Cog."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.cogs import admin
from app.core.bot import DisBot


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
