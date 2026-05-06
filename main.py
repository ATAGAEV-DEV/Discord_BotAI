import os

from dotenv import load_dotenv

load_dotenv()

import discord  # noqa: E402

from app.core.bot import DisBot  # noqa: E402

TOKEN = os.getenv("DC_TOKEN")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True


# Лимиты
CONTEXT_LIMIT = 100  # Количество строк контекста для RAG
REPORT_MSG_LIMIT = 15  # Порог сообщений для создания отчета
REPORT_TIME_LIMIT = 60  # Время ожидания в минутах для создания отчета


def main() -> None:
    """Запуск бота."""
    bot = DisBot(
        command_prefix="!",
        intents=intents,
        context_limit=CONTEXT_LIMIT,
        report_msg_limit=REPORT_MSG_LIMIT,
        report_time_limit=REPORT_TIME_LIMIT,
        help_command=None,
    )

    if not TOKEN:
        print("Ошибка: DC_TOKEN не найден в переменных окружения!")
        return

    bot.run(TOKEN)


if __name__ == "__main__":
    main()
