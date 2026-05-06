from discord.ext import commands

from app.core.bot import DisBot
from app.core.checks import admin_or_owner


class YouTube(commands.Cog):
    """Команды YouTube-уведомлений."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot
        self.youtube_notifier = bot.youtube_notifier

    @commands.command(name="add_youtube")
    @commands.guild_only()
    @admin_or_owner()
    async def add_youtube_command(
        self, ctx: commands.Context, youtube_id: str, discord_channel_id: int, *, name: str
    ) -> None:
        """Добавить YouTube канал для отслеживания."""
        try:
            channel = self.bot.get_channel(discord_channel_id)
            if channel is None:
                await ctx.send("❌ Канал не найден!")
                return

            success = await self.youtube_notifier.add_channel(
                youtube_id, discord_channel_id, name, ctx.guild.id
            )
            if success:
                await ctx.send("✅ Канал добавлен для отслеживания")
            else:
                await ctx.send("❌ Ошибка при добавлении канала")
        except Exception as e:
            await ctx.send(f"❌ Ошибка: {e}")

    @commands.command(name="youtube")
    @commands.guild_only()
    @admin_or_owner()
    async def youtube_toggle_command(self, ctx: commands.Context, action: str, *, name: str) -> None:
        """Включить или отключить отслеживание YouTube канала."""
        action = action.lower()
        if action not in ("on", "off"):
            await ctx.send("❌ Используйте: `!youtube on/off название_канала`")
            return

        active = action == "on"
        result = await self.youtube_notifier.toggle_channel(name, ctx.guild.id, active)

        if result is None:
            await ctx.send(f"❌ Канал **{name}** не найден на этом сервере.")
        elif result:
            status = "включено" if active else "отключено"
            emoji = "✅" if active else "⏸️"
            await ctx.send(f"{emoji} Отслеживание канала **{name}** {status}.")
        else:
            await ctx.send("❌ Ошибка при изменении статуса канала.")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(YouTube(bot))
