import os
import sys

print("Начинаем проверку зависимостей...")
try:
    import aiohttp  # noqa: F401
    import apscheduler  # noqa: F401
    import asyncpg  # noqa: F401
    import chromadb  # noqa: F401
    import discord
    import llama_index.core  # noqa: F401
    import openai  # noqa: F401
    import PIL  # noqa: F401
    import pytz  # noqa: F401
    import sqlalchemy  # noqa: F401
    import tiktoken  # noqa: F401
    from dotenv import load_dotenv  # noqa: F401

    print("Все зависимости успешно загружены.")
except ImportError as e:
    print(f"Импорт не удалось: {e}")
    sys.exit(1)

load_dotenv()
DC_TOKEN_TEST = os.getenv("DC_TOKEN_TEST")

if not DC_TOKEN_TEST:
    print("DC_TOKEN_TEST не найден в переменных окружения. Пропускаем тест подключения к Discord.")
    print("Предупреждение: DC_TOKEN_TEST отсутствует!")
    sys.exit(1)
else:
    print("Начинаем тест подключения к Discord...")

    class TestClient(discord.Client):
        """Тестовый клиент Discord для проверки подключения."""

        async def on_ready(self) -> None:
            """Обработчик успешного подключения к Discord."""
            if self.user:
                print(f"Вход выполнен как {self.user} (ID: {self.user.id})")
            print("Тест подключения к Discord прошел успешно.")
            await self.close()

    intents = discord.Intents.default()
    client = TestClient(intents=intents)

    try:
        client.run(DC_TOKEN_TEST)
    except Exception as e:
        print(f"Подключение к Discord не удалось: {e}")
        sys.exit(1)

print("Тест скрипта завершен.")
