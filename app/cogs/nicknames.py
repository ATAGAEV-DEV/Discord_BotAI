"""Команды управления описаниями пользователей."""

from discord.ext import commands

from app.core.bot import DisBot
from app.core.checks import admin_or_owner
from app.data import user_descriptions_cache


class Nicknames(commands.Cog):
    """Управление описаниями пользователей сервера."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot

    @commands.command(name="desc_add")
    @commands.guild_only()
    @admin_or_owner()
    async def desc_add_command(
        self, ctx: commands.Context, nick: str, *, description: str
    ) -> None:
        """Добавить или обновить описание пользователя.

        Использование: !desc_add ник Описание пользователя
        Пример: !desc_add atagaev Арби, создатель бота
        """
        try:
            result = await user_descriptions_cache.save(
                nick=nick,
                description=description,
                guild_id=ctx.guild.id,
            )
            await ctx.send(f"✅ {result}")
        except Exception as e:
            await ctx.send(f"❌ Ошибка: {e}")

    @commands.command(name="desc_remove")
    @commands.guild_only()
    @admin_or_owner()
    async def desc_remove_command(self, ctx: commands.Context, nick: str) -> None:
        """Удалить описание пользователя.

        Использование: !desc_remove ник
        Пример: !desc_remove atagaev
        """
        try:
            result = await user_descriptions_cache.remove(
                nick=nick,
                guild_id=ctx.guild.id,
            )
            await ctx.send(f"✅ {result}")
        except Exception as e:
            await ctx.send(f"❌ Ошибка: {e}")

    @commands.command(name="desc_list")
    @commands.guild_only()
    async def desc_list_command(self, ctx: commands.Context) -> None:
        """Показать все описания пользователей сервера.

        Использование: !desc_list
        """
        descriptions = user_descriptions_cache.get(ctx.guild.id)

        if not descriptions:
            await ctx.send("📭 Описания пользователей не найдены.")
            return

        lines = [f"**{nick}** — {desc}" for nick, desc in descriptions.items()]
        text = "📋 **Описания пользователей:**\n" + "\n".join(lines)
        await ctx.send(text)

    @commands.command(name="desc_reload")
    @commands.guild_only()
    @admin_or_owner()
    async def desc_reload_command(self, ctx: commands.Context) -> None:
        """Перезагрузить кэш описаний из БД.

        Использование: !desc_reload
        """
        try:
            await user_descriptions_cache.load_all()
            count = len(user_descriptions_cache.get(ctx.guild.id))
            await ctx.send(
                f"🔄 Кэш описаний перезагружен! "
                f"Загружено {count} описаний для этого сервера."
            )
        except Exception as e:
            await ctx.send(f"❌ Ошибка перезагрузки: {e}")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(Nicknames(bot))
