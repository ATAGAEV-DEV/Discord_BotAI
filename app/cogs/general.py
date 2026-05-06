from discord.ext import commands

import app.core.embeds as em
from app.core.bot import DisBot
from app.data.request import get_rank, save_birthday
from app.tools.utils import get_rank_description, parse_birthday_date


class General(commands.Cog):
    """Общие команды бота."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot

    @commands.command(name="help")
    async def help_command(self, ctx: commands.Context) -> None:
        """Показать список команд."""
        embed = em.create_help_embed()
        await ctx.send(embed=embed)

    @commands.command(name="rank")
    @commands.cooldown(rate=1, per=10.0, type=commands.BucketType.user)
    async def rank_command(self, ctx: commands.Context, arg: str | None = None) -> None:
        """Показать ранг пользователя или список рангов.

        Если указан аргумент "list", отображает список всех рангов.
        В противном случае показывает текущий ранг автора сообщения.
        """
        if arg == "list":
            embed = em.create_rang_list_embed()
            await ctx.send(embed=embed)
            return

        try:
            async with ctx.typing():
                server_id = ctx.guild.id if ctx.guild else None
                message_count = await get_rank(ctx.author.id, server_id)
                rank_description = get_rank_description(int(message_count))

                avatar_url = (
                    ctx.author.avatar.url if ctx.author.avatar else ctx.author.default_avatar.url
                )

                embed, file = await em.create_rang_embed(
                    ctx.author.display_name,
                    message_count,
                    rank_description["description"],
                    avatar_url,
                    server_id,
                    ctx.author.id,
                )
                await ctx.send(embed=embed, file=file)
        except ValueError as ve:
            await ctx.send(str(ve))
        except Exception as e:
            await ctx.send(f"Произошла ошибка при получении статистики: {e}")

    @commands.command(name="birthday")
    @commands.cooldown(rate=1, per=30.0, type=commands.BucketType.user)
    async def birthday_command(self, ctx: commands.Context, *, date: str) -> None:
        """Сохранить дату рождения."""
        try:
            birthday = parse_birthday_date(date)
            await save_birthday(
                ctx.author.id,
                ctx.author.display_name,
                ctx.author.name,
                birthday,
            )
            await ctx.send("Дата рождения сохранена.")
        except ValueError as ve:
            await ctx.send(str(ve))
        except Exception as e:
            await ctx.send(f"Произошла ошибка при сохранении даты рождения: {e}")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(General(bot))
