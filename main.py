import os

import discord
from dotenv import load_dotenv

load_dotenv()

from app.core.bot import DisBot  # noqa: E402
from app.core.config import get_command_prefix  # noqa: E402

# Лимиты
CONTEXT_LIMIT = 100  # Количество строк контекста для RAG
REPORT_MSG_LIMIT = 15  # Порог сообщений для создания отчета
REPORT_TIME_LIMIT = 60  # Время ожидания в минутах для создания отчета


def main() -> None:
    """Запуск бота."""
    token = os.getenv("DC_TOKEN")
    if not token:
        print("Ошибка: DC_TOKEN не найден в переменных окружения!")
        return

    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True

    bot = DisBot(
        command_prefix=get_command_prefix(),
        intents=intents,
        context_limit=CONTEXT_LIMIT,
        report_msg_limit=REPORT_MSG_LIMIT,
        report_time_limit=REPORT_TIME_LIMIT,
        help_command=None,
    )

    bot.run(token)


if __name__ == "__main__":
    main()
