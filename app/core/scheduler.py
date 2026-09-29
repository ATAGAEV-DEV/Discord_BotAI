import discord
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import get_scheduler_settings
from app.services.youtube_notifier import YouTubeNotifier


def start_scheduler(bot: discord.Client, youtube_notifier: YouTubeNotifier) -> None:
    """Инициализирует и запускает асинхронный планировщик задач."""
    settings = get_scheduler_settings()
    scheduler = AsyncIOScheduler(timezone=pytz.timezone(settings.timezone))

    scheduler.add_job(
        youtube_notifier.check_new_videos,
        "interval",
        minutes=settings.youtube_check_interval_minutes,
        id="youtube_check",
    )
    scheduler.start()
