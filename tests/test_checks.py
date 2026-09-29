"""Тесты проверки прав администратора."""

from unittest.mock import MagicMock, patch

import pytest
from discord.ext import commands

from app.core.checks import admin_or_owner


def make_context(*, is_discord_admin: bool, guild_id: int = 67890) -> MagicMock:
    """Создаёт минимальный контекст команды для проверки прав."""
    ctx = MagicMock(spec=commands.Context)
    ctx.guild.id = guild_id
    ctx.author.id = 12345
    ctx.author.guild_permissions.administrator = is_discord_admin
    return ctx


@pytest.mark.asyncio
async def test_discord_administrator_is_allowed() -> None:
    """Администратор Discord получает доступ без записи в БД."""
    ctx = make_context(is_discord_admin=True)

    result = await admin_or_owner().predicate(ctx)

    assert result is True


@pytest.mark.asyncio
@patch("app.core.checks.admins.is_admin", return_value=True)
async def test_database_administrator_is_allowed(mock_is_admin: MagicMock) -> None:
    """Администратор из БД получает доступ."""
    ctx = make_context(is_discord_admin=False)

    result = await admin_or_owner().predicate(ctx)

    assert result is True
    mock_is_admin.assert_called_once_with(67890, 12345)


@pytest.mark.asyncio
@patch("app.core.checks.admins.is_admin", return_value=False)
async def test_regular_user_is_denied(mock_is_admin: MagicMock) -> None:
    """Пользователь без прав получает ошибку доступа."""
    ctx = make_context(is_discord_admin=False)

    with pytest.raises(commands.MissingPermissions):
        await admin_or_owner().predicate(ctx)

    mock_is_admin.assert_called_once_with(67890, 12345)
