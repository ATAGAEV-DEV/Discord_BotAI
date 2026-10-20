"""Команды управления справочником AI-моделей."""

from __future__ import annotations

from discord.ext import commands

from app.core.bot import DisBot
from app.core.checks import admin_or_owner
from app.data.models import async_session
from app.data.repositories import ai_models
from app.data.repositories.guild_ai_config import GuildAIConfigError, update_model


class AIModels(commands.Cog):
    """Управление списком AI-моделей и выбор модели для сервера."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot

    @commands.command(name="model_list")
    @commands.guild_only()
    @admin_or_owner()
    async def model_list_command(self, ctx: commands.Context) -> None:
        """Показать список всех доступных AI-моделей."""
        async with async_session() as session:
            models = await ai_models.get_all(session)

        if not models:
            await ctx.send("📭 Список моделей пуст.")
            return

        lines = [f"{i}. `{name}`" for i, name in enumerate(models, start=1)]
        await ctx.send("🤖 **Список AI-моделей:**\n" + "\n".join(lines))

    @commands.command(name="model_add")
    @commands.guild_only()
    @admin_or_owner()
    async def model_add_command(self, ctx: commands.Context, *, name: str) -> None:
        """Добавить модель в список. Использование: !model_add <имя>"""
        async with async_session() as session:
            result = await ai_models.add(session, name)
        await ctx.send(result)

    @commands.command(name="model_remove")
    @commands.guild_only()
    @admin_or_owner()
    async def model_remove_command(self, ctx: commands.Context, *, name: str) -> None:
        """Убрать модель из списка. Использование: !model_remove <имя>"""
        async with async_session() as session:
            result = await ai_models.remove(session, name)
        await ctx.send(result)

    @commands.command(name="model_set")
    @commands.guild_only()
    @admin_or_owner()
    async def model_set_command(
        self, ctx: commands.Context, model_type: str, *, name: str
    ) -> None:
        """Выбрать модель для сервера. Использование: !model_set <chat|mini> <имя>"""
        model_type = model_type.lower().strip()
        if model_type not in {"chat", "mini"}:
            await ctx.send("❌ Тип модели должен быть `chat` или `mini`.")
            return

        field = "chat_model" if model_type == "chat" else "mini_model"

        async with async_session() as session:
            if not await ai_models.exists(session, name):
                await ctx.send(f"❌ Модель `{name}` не найдена в списке. Сначала добавьте её через `!model_add`.")
                return
            try:
                await update_model(session, ctx.guild.id, field, name)
            except GuildAIConfigError as exc:
                await ctx.send(f"❌ {exc}")
                return

        label = "основная модель" if field == "chat_model" else "мини-модель"
        await ctx.send(f"✅ {label.capitalize()} для сервера установлена: `{name}`.")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(AIModels(bot))

