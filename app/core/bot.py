import discord
from discord.ext import commands

from app.core.scheduler import start_scheduler
from app.data import user_descriptions_cache
from app.data.models import init_models
from app.services.daily_report import ReportGenerator
from app.services.youtube_notifier import YouTubeNotifier
from app.tools.utils import contains_only_urls


class DisBot(commands.Bot):
    """Кастомный класс бота Discord."""

    def __init__(
        self,
        command_prefix: str,
        intents: discord.Intents,
        context_limit: int = 50,
        report_msg_limit: int = 15,
        report_time_limit: int = 60,
        help_command: commands.HelpCommand | None = None,
    ):
        """Инициализация бота."""
        super().__init__(command_prefix=command_prefix, intents=intents, help_command=help_command)
        self.report_generator: ReportGenerator | None = None
        self.youtube_notifier: YouTubeNotifier = YouTubeNotifier(self)

        self.context_limit: int = context_limit
        self.report_msg_limit: int = report_msg_limit
        self.report_time_limit: int = report_time_limit

    async def setup_hook(self) -> None:
        """Загрузка расширений (Cogs) при старте бота."""
        await self.load_extension("app.cogs.general")
        await self.load_extension("app.cogs.admin")
        await self.load_extension("app.cogs.youtube")
        await self.load_extension("app.cogs.toxic")
        await self.load_extension("app.cogs.nicknames")
        await self.load_extension("app.cogs.error_handler")
        await self.load_extension("app.cogs.ranks")
        start_scheduler(self, self.youtube_notifier)

    async def on_ready(self) -> None:
        """Инициализация при подключении бота к Discord."""
        await init_models()
        await user_descriptions_cache.load_all()
        self.report_generator = ReportGenerator(self)

        print("Бот успешно подключился к Discord")

    async def on_disconnect(self) -> None:
        """Обработка отключения от Discord."""
        print("Бот отключился от Discord")

    async def on_resumed(self) -> None:
        """Обработка восстановления соединения с Discord."""
        print("Соединение с Discord восстановлено")

    async def on_message(self, message: discord.Message) -> None:
        """Обработка входящих сообщений."""
        if message.author.bot:
            return

        if len(message.content) > 1000:
            if message.content.startswith(self.command_prefix):
                await message.channel.send(
                    f"Сообщение слишком длинное: {len(message.content)} символов! "
                    "Максимальная длина - 1000 символов."
                )
            return

        if not message.content.startswith(self.command_prefix):
            if not message.content.strip():
                return

            if contains_only_urls(message.content):
                return

            if self.report_generator is not None and not message.content.startswith("?"):
                await self.report_generator.add_message(
                    message.channel.id,
                    message.content,
                    message.author.display_name,
                    message.id,
                )
            return

        await self.process_commands(message)
