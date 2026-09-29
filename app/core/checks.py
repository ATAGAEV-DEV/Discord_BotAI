from collections.abc import Callable

from discord.ext import commands

from app.data import admins


def admin_or_owner() -> Callable:
    """Проверка: администратор Discord или администратор из БД."""

    async def predicate(ctx: commands.Context) -> bool:
        if ctx.author.guild_permissions.administrator:
            return True
        if ctx.guild is not None and admins.is_admin(ctx.guild.id, ctx.author.id):
            return True
        raise commands.MissingPermissions(["administrator"])

    return commands.check(predicate)
