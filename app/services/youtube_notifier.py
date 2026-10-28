import asyncio
from datetime import datetime
from typing import Any

import feedparser
import pytz
from sqlalchemy import select, update
from sqlalchemy.exc import MultipleResultsFound

from app.core import config
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
                    try:
                        await self._check_channel_videos(channel)
                    # Любая ошибка канала не должна прерывать проверку остальных.
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"Ошибка при проверке YouTube канала "
                            f"{channel.name} (ID: {channel.channel_id}): {e}"
                        )
        except Exception as e:
            print(f"Ошибка при проверке YouTube видео: {e}")

    async def _check_channel_videos(self, channel: Any) -> None:
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel.channel_id}"
        feed = await asyncio.to_thread(feedparser.parse, url)

        # При сетевом сбое feedparser может вернуть результат без HTTP-статуса.
        status = getattr(feed, "status", None)
        entries = getattr(feed, "entries", [])
        if status != 200 or not entries:
            timestamp = datetime.now(tz=pytz.timezone(config.BOT_TIMEZONE)).strftime(
                "%d.%m.%Y %H:%M:%S"
            )
            description = (
                "Не удалось загрузить RSS канала"
                if status is None
                else "Неверный или недоступный канал"
            )
            print(f"[{timestamp}] ❌ {description}: {channel.name} (ID: {channel.channel_id})")
            print(f"HTTP статус: {status if status is not None else 'отсутствует'}")
            error = getattr(feed, "bozo_exception", None)
            if error is not None:
                print(f"Причина: {error}")
            return

        latest_video = entries[0]
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

                # ID в двух БД могут различаться; обновляем все копии одной подписки.
                await session.execute(
                    update(YouTubeChannel)
                    .where(
                        YouTubeChannel.channel_id == channel.channel_id,
                        YouTubeChannel.discord_channel_id == channel.discord_channel_id,
                        YouTubeChannel.guild_id == channel.guild_id,
                    )
                    .values(is_active=active)
                )
                await session.commit()
                return True
        except MultipleResultsFound:
            print(f"Неоднозначное имя YouTube канала: {name}")
            return None
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
