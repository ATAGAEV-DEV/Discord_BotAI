import asyncio
from datetime import datetime
from typing import Any

import feedparser
import pytz
from sqlalchemy import select

from app.data.models import YouTubeChannel, YouTubeVideo, async_session


class YouTubeNotifier:
    """Класс для отслеживания новых видео на YouTube."""

    def __init__(self, bot: Any) -> None:
        """Инициализирует YouTube-нотифайер."""
        self.bot = bot

    async def check_new_videos(self) -> None:
        """Проверяет все отслеживаемые YouTube-каналы на наличие новых видео."""
        try:
            async with async_session() as session:
                query = select(YouTubeChannel).where(YouTubeChannel.is_active.is_(True))
                result = await session.execute(query)
                channels = result.scalars().all()

                for channel in channels:
                    await self._check_channel_videos(channel)
        except Exception as e:
            print(f"Ошибка при проверке YouTube видео: {e}")

    async def _check_channel_videos(self, channel: Any) -> None:
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel.channel_id}"
        feed = await asyncio.to_thread(feedparser.parse, url)

        if feed.status != 200 or not feed.entries:
            timestamp = datetime.now(tz=pytz.timezone("Europe/Moscow")).strftime("%d.%m.%Y %H:%M:%S")
            print(
                f"[{timestamp}] ❌ Неверный или недоступный канал: "
                f"{channel.name} (ID: {channel.channel_id})"
            )
            print(f"HTTP статус: {feed.status}")
            return

        latest_video = feed.entries[0]
        video_id = latest_video.get("yt_videoid")
        author = latest_video.get("author")
        published_at = datetime.now(tz=pytz.utc).replace(tzinfo=None)

        async with async_session() as session:
            video_query = select(YouTubeVideo).where(
                YouTubeVideo.video_id == video_id, YouTubeVideo.guild_id == channel.guild_id
            )
            video_result = await session.execute(video_query)
            existing_video = video_result.scalar_one_or_none()

            if not existing_video:
                is_live = "Live" in latest_video.title or "прямая" in latest_video.title.lower()
                new_video = YouTubeVideo(
                    video_id=video_id,
                    guild_id=channel.guild_id,
                    channel_id=channel.channel_id,
                    title=latest_video.title,
                    published_at=published_at,
                    is_live=is_live,
                )
                session.add(new_video)

                discord_channel = self.bot.get_channel(channel.discord_channel_id)
                if discord_channel:
                    if is_live:
                        message = (
                            f"🔴 **Прямой эфир на канале [{author}](https://www.youtube.com/channel/{channel.channel_id})!**\n"
                            f"{latest_video.link}"
                        )
                    else:
                        message = (
                            f"🎥 **Новое видео на канале [{author}](https://www.youtube.com/channel/{channel.channel_id})!**\n"
                            f"{latest_video.link}"
                        )
                    await discord_channel.send(message)

            await session.commit()

    async def toggle_channel(self, name: str, guild_id: int, active: bool) -> bool | None:
        """Переключает статус отслеживания YouTube-канала."""
        try:
            async with async_session() as session:
                query = select(YouTubeChannel).where(
                    YouTubeChannel.name == name,
                    YouTubeChannel.guild_id == guild_id,
                )
                result = await session.execute(query)
                channel = result.scalar_one_or_none()

                if not channel:
                    return None

                channel.is_active = active
                await session.commit()
                return True
        except Exception as e:
            print(f"Ошибка при переключении YouTube канала: {e}")
            return False

    async def add_channel(
        self, youtube_channel_id: str, discord_channel_id: int, name: str, guild_id: int
    ) -> bool | None:
        """Добавляет новый YouTube-канал для отслеживания."""
        try:
            async with async_session() as session:
                query = select(YouTubeChannel).where(
                    YouTubeChannel.channel_id == youtube_channel_id,
                    YouTubeChannel.discord_channel_id == discord_channel_id,
                    YouTubeChannel.guild_id == guild_id,
                )
                result = await session.execute(query)
                existing_channel = result.scalar_one_or_none()

                if existing_channel:
                    return False

                channel = YouTubeChannel(
                    channel_id=youtube_channel_id,
                    discord_channel_id=discord_channel_id,
                    name=name,
                    guild_id=guild_id,
                )
                session.add(channel)
                await session.commit()
                return True
        except Exception as e:
            print(f"Ошибка при добавлении YouTube канала: {e}")
            return False
