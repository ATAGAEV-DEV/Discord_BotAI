import os

import discord
from dotenv import load_dotenv

load_dotenv()

from app.core import config  # noqa: E402
from app.core.bot import DisBot  # noqa: E402


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
        command_prefix=config.get_command_prefix(),
        intents=intents,
        context_limit=config.CONTEXT_LIMIT,
        report_msg_limit=config.REPORT_MSG_LIMIT,
        report_time_limit=config.REPORT_TIME_LIMIT,
        help_command=None,
    )

    bot.run(token)


if __name__ == "__main__":
    main()
