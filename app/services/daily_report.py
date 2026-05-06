import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytz
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.core.ai_config import get_client, get_mini_model
from app.data.request import delete_channel_messages, get_channel_messages, save_channel_message
from app.tools.prompt import UPDATED_REPORT_PROMPT


@dataclass
class ChannelState:
    """Состояние канала для генерации отчетов."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_message_time: datetime = field(default_factory=lambda: datetime.now(tz=pytz.utc))
    timer: asyncio.Task | None = None


class ReportGenerator:
    """Генератор отчётов по активности в Discord-каналах.

    Накапливает сообщения и автоматически отправляет отчёт
    после заданного времени без активности при достижении порогового
    количества сообщений.
    """

    def __init__(self, bot: Any) -> None:
        """Инициализирует генератор отчётов."""
        self.bot = bot
        self.channels: dict[int, ChannelState] = {}

    def get_state(self, channel_id: int) -> ChannelState:
        """Возвращает (или создает) состояние для указанного канала."""
        if channel_id not in self.channels:
            self.channels[channel_id] = ChannelState()
        return self.channels[channel_id]

    async def add_message(self, channel_id: int, message: str, author: str, message_id: int) -> None:
        """Добавляет сообщение в историю канала.

        Обновляет время последней активности. При достижении или
        превышении лимита сообщений запускает задачу отправки отчёта.
        """
        state = self.get_state(channel_id)
        async with state.lock:
            try:
                await save_channel_message(channel_id, message_id, author, message)
            except Exception as e:
                print(f"Ошибка при сохранении сообщения в канал {channel_id}: {e}")

            state.messages.append(
                {
                    "id": message_id,
                    "content": message,
                    "author": author,
                    "timestamp": datetime.now(tz=pytz.utc),
                }
            )

            state.last_message_time = datetime.now(tz=pytz.utc)

            # Если таймер уже запущен — отменяем, так как активность продолжилась
            if state.timer and not state.timer.done():
                state.timer.cancel()

            try:
                db_messages = await get_channel_messages(channel_id)
            except Exception as e:
                print(f"Ошибка при получении сообщений из канала {channel_id}: {e}")
                db_messages = []

            cache_count = len(state.messages)
            db_count = len(db_messages)

            if cache_count >= self.bot.report_msg_limit or db_count >= self.bot.report_msg_limit:
                state.timer = asyncio.create_task(self.start_report_timer(channel_id))

    async def start_report_timer(self, channel_id: int) -> None:
        """Запускает таймер ожидания.

        Если после последнего сообщения прошло достаточно времени,
        генерирует и отправляет аналитический отчёт.
        """
        await asyncio.sleep(self.bot.report_time_limit * 60)

        if channel_id not in self.channels:
            return

        state = self.channels[channel_id]
        async with state.lock:
            if (datetime.now(tz=pytz.utc) - state.last_message_time) < timedelta(
                minutes=self.bot.report_time_limit
            ):
                return

        await self.generate_and_send_report(channel_id)

    async def generate_and_send_report(self, channel_id: int) -> None:
        """Генерирует аналитический отчёт и отправляет его в канал.

        Использует GPT-модель для анализа накопленных сообщений.
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            print(f"Канал {channel_id} недоступен, пропускаем генерацию отчёта")
            self.channels.pop(channel_id, None)
            return

        state = self.get_state(channel_id)
        async with state.lock:
            try:
                messages = await get_channel_messages(channel_id)
            except Exception as e:
                print(
                    f"Ошибка при получении сообщений для генерации отчета в канале {channel_id}: {e}"
                )
                return

            if not messages or len(messages) < self.bot.report_msg_limit:
                return

            messages_text = "\n".join(
                f"[ID:{msg.message_id}] {msg.author}: {msg.content}" for msg in messages
            )

            message_payload = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=UPDATED_REPORT_PROMPT,
                ),
                ChatCompletionUserMessageParam(
                    role="user", content=f"Сообщения из канала:\n{messages_text}"
                ),
            ]

            try:
                response = await get_client().chat.completions.create(
                    model=get_mini_model(),
                    messages=message_payload,
                    temperature=0.0,
                    top_p=0.01,
                )
                report = response.choices[0].message.content or "Пустой ответ от AI"
            except Exception as e:
                print(f"Ошибка генерации отчета: {e}")
                return

            if "ID:" in report:
                guild_id = channel.guild.id if channel and hasattr(channel, "guild") else "UNKNOWN"

                for msg in messages:
                    msg_id = str(msg.message_id)
                    if f"[ID:{msg_id}]" in report:
                        link = f"https://discord.com/channels/{guild_id}/{channel_id}/{msg_id}"
                        report = report.replace(f"[ID:{msg_id}]", f"[ссылка]({link})")

            try:
                if channel:
                    await channel.send(report)
            except Exception as e:
                print(f"Ошибка отправки отчета в канал {channel_id}: {e}")
                return

            try:
                await delete_channel_messages(channel_id)
            except Exception as e:
                print(f"Ошибка при удалении сообщений из канала {channel_id}: {e}")

            if channel_id in self.channels:
                del self.channels[channel_id]
