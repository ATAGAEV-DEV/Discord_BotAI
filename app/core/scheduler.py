import discord
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.youtube_notifier import YouTubeNotifier


def start_scheduler(bot: discord.Client, youtube_notifier: YouTubeNotifier) -> None:
    """Инициализирует и запускает асинхронный планировщик задач."""
    scheduler = AsyncIOScheduler(timezone=pytz.timezone("Europe/Moscow"))

    scheduler.add_job(youtube_notifier.check_new_videos, "interval", minutes=5, id="youtube_check")
    scheduler.start()
