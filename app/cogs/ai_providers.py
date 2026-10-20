"""Команды управления справочником AI-провайдеров."""

from __future__ import annotations

from discord.ext import commands

from app.core.bot import DisBot
from app.core.checks import admin_or_owner
from app.data.models import async_session
from app.data.repositories import ai_providers
from app.data.repositories.guild_ai_config import GuildAIConfigError, update_provider


class AIProviders(commands.Cog):
    """Управление списком AI-провайдеров и выбор провайдера для сервера."""

    def __init__(self, bot: DisBot) -> None:
        """Инициализация Cog."""
        self.bot = bot

    @commands.command(name="provider_list")
    @commands.guild_only()
    @admin_or_owner()
    async def provider_list_command(self, ctx: commands.Context) -> None:
        """Показать список всех доступных AI-провайдеров."""
        async with async_session() as session:
            providers = await ai_providers.get_all(session)

        if not providers:
            await ctx.send("📭 Список провайдеров пуст.")
            return

        lines = [f"{i}. `{name}` — {url}" for i, (name, url) in enumerate(providers, start=1)]
        await ctx.send("🌐 **Список AI-провайдеров:**\n" + "\n".join(lines))

    @commands.command(name="provider_add")
    @commands.guild_only()
    @admin_or_owner()
    async def provider_add_command(
        self, ctx: commands.Context, name: str, url: str, *, token: str
    ) -> None:
        """Добавить провайдера. Использование: !provider_add <имя> <url> <токен>."""
        async with async_session() as session:
            result = await ai_providers.add(session, name, url, token)
        await ctx.send(result)

    @commands.command(name="provider_remove")
    @commands.guild_only()
    @admin_or_owner()
    async def provider_remove_command(self, ctx: commands.Context, *, name: str) -> None:
        """Убрать провайдера из списка. Использование: !provider_remove <имя>."""
        async with async_session() as session:
            result = await ai_providers.remove(session, name)
        await ctx.send(result)

    @commands.command(name="provider_set")
    @commands.guild_only()
    @admin_or_owner()
    async def provider_set_command(
        self, ctx: commands.Context, provider_type: str, *, name: str
    ) -> None:
        """Выбрать провайдера для сервера. Использование: !provider_set <chat|embedding> <имя>."""
        provider_type = provider_type.lower().strip()
        if provider_type not in {"chat", "embedding"}:
            await ctx.send("❌ Тип провайдера должен быть `chat` или `embedding`.")
            return

        async with async_session() as session:
            provider = await ai_providers.get(session, name)
            if provider is None:
                await ctx.send(
                    f"❌ Провайдер `{name}` не найден в списке."
                    " Сначала добавьте его через `!provider_add`."
                )
                return
            try:
                await update_provider(
                    session,
                    ctx.guild.id,
                    provider_type,
                    provider.name,
                    provider.api_url,
                    provider.token,
                )
            except GuildAIConfigError as exc:
                await ctx.send(f"❌ {exc}")
                return

        label = "основной провайдер" if provider_type == "chat" else "провайдер эмбеддингов"
        await ctx.send(f"✅ {label.capitalize()} для сервера установлен: `{name}`.")


async def setup(bot: DisBot) -> None:
    """Загрузка Cog в бота."""
    await bot.add_cog(AIProviders(bot))
