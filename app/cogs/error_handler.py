import time
from collections import defaultdict

from discord.ext import commands

from app.core import handlers
from app.core.bot import DisBot

AI_COOLDOWN_SECONDS = 5.0


class ErrorHandler(commands.Cog):
    """Обработчик ошибок команд."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot
        self._ai_cooldowns: dict[int, float] = defaultdict(float)

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        """Обработка ошибок команд."""
        original_error = getattr(error, "original", error)

        if isinstance(error, commands.CommandNotFound):
            now = time.monotonic()
            last_use = self._ai_cooldowns[ctx.author.id]
            if now - last_use < AI_COOLDOWN_SECONDS:
                remaining = AI_COOLDOWN_SECONDS - (now - last_use)
                await ctx.send(
                    f"⏳ {ctx.author.mention}, подождите {remaining:.0f} сек. "
                    "перед следующим сообщением."
                )
                return
            self._ai_cooldowns[ctx.author.id] = now

            server_id = ctx.guild.id if ctx.guild else None

            async with ctx.typing():
                response = await handlers.ai_generate(
                    ctx.message.content,
                    server_id,
                    ctx.author,
                    limit=self.bot.context_limit,
                )
                await ctx.send(f"{ctx.author.mention} {response}")
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.send(
                f"⏳ Подождите {error.retry_after:.0f} сек. перед повторным использованием."
            )
        elif isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ У вас недостаточно прав для выполнения этой команды.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                f"❌ Неправильное использование команды. "
                f"Используйте: `{self.bot.command_prefix}{ctx.command.name} "
                f"{ctx.command.signature}`"
            )
        elif isinstance(error, commands.NoPrivateMessage):
            await ctx.send("❌ Эта команда недоступна в личных сообщениях.")
        elif isinstance(original_error, (ConnectionError, TimeoutError)):
            await ctx.send("❌ Проблема с сетью. Попробуйте позже.")

        else:
            await ctx.send("❌ Произошла ошибка при выполнении команды.")
            print(f"Command error: {error}")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(ErrorHandler(bot))
