from discord.ext import commands

from app.core import handlers
from app.core.ai_config import (
    get_active_provider,
    get_available_providers,
    next_provider,
    set_active_provider,
)
from app.core.bot import DisBot
from app.core.checks import admin_or_owner
from app.data import admins


class Admin(commands.Cog):
    """Административные команды."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot

    @commands.command(name="admin_add")
    @commands.guild_only()
    @admin_or_owner()
    async def admin_add_command(self, ctx: commands.Context, name: str, user_id: int) -> None:
        """Добавить администратора бота для текущего сервера."""
        result = await admins.add(
            guild_id=ctx.guild.id,
            user_id=user_id,
            username=name,
        )
        await ctx.send(result)

    @commands.command(name="admin_remove")
    @commands.guild_only()
    @admin_or_owner()
    async def admin_remove_command(self, ctx: commands.Context, user_id: int) -> None:
        """Удалить администратора бота текущего сервера."""
        result = await admins.remove(guild_id=ctx.guild.id, user_id=user_id)
        await ctx.send(result)

    @commands.command(name="admin_list")
    @commands.guild_only()
    @admin_or_owner()
    async def admin_list_command(self, ctx: commands.Context) -> None:
        """Показать администраторов бота текущего сервера."""
        server_admins = admins.get_all(ctx.guild.id)
        if not server_admins:
            await ctx.send("📭 Администраторы бота для этого сервера не назначены.")
            return

        lines = [
            f"{index}. **{username}** — `{user_id}`"
            for index, (user_id, username) in enumerate(server_admins.items(), start=1)
        ]
        await ctx.send("🛡️ **Администраторы бота:**\n" + "\n".join(lines))

    @commands.command(name="reset")
    @commands.guild_only()
    @admin_or_owner()
    async def reset_command(self, ctx: commands.Context) -> None:
        """Очистить историю сервера (только для администраторов)."""
        async with ctx.typing():
            answer = await handlers.clear_server_history(ctx.guild.id)
            await ctx.send(answer)

    @commands.command(name="update_user")
    @commands.guild_only()
    @admin_or_owner()
    async def update_user_command(self, ctx: commands.Context) -> None:
        """Обновить список пользователей сервера."""
        server = ctx.guild
        members = server.members
        all_server_users = [f"{member.name}" for member in members if not member.bot]

        await handlers.llama_manager.index_server_users(server.id, all_server_users)

        await ctx.send(
            f"✅ Список пользователей сервера обновлен! "
            f"Добавлено {len(all_server_users)} пользователей."
        )

    @commands.command(name="ai")
    @admin_or_owner()
    async def ai_provider_command(self, ctx: commands.Context, name: str | None = None) -> None:
        """Переключить AI-провайдера.

        !ai — переключить на следующего провайдера
        !ai name — переключить на указанного провайдера
        """
        available = get_available_providers()

        if name is None:
            new_provider = next_provider()
            await ctx.send(
                f"🔄 Провайдер переключён на **{new_provider}**\nДоступные: {', '.join(available)}"
            )
            return

        if name not in available:
            await ctx.send(f"❌ Провайдер **{name}** не найден.\nДоступные: {', '.join(available)}")
            return

        if name == get_active_provider():
            await ctx.send(f"ℹ️ Провайдер **{name}** уже активен.")
            return

        set_active_provider(name)
        await ctx.send(f"✅ Провайдер переключён на **{name}**")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(Admin(bot))
