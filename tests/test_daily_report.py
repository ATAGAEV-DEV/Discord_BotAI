"""Unit-тесты для app/services/daily_report.py."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytz

from app.services import daily_report
from app.services.daily_report import ChannelState, ReportGenerator


def _message(message_id: int, author: str = "user", content: str = "message") -> SimpleNamespace:
    """Создаёт объект сообщения с атрибутами модели ChannelMessage."""
    return SimpleNamespace(message_id=message_id, author=author, content=content)


class TestReportGeneratorGetState:
    """Тесты метода get_state."""

    def test_creates_new_state(self) -> None:
        """Создает новый ChannelState для нового канала."""
        bot = MagicMock()
        rg = ReportGenerator(bot=bot)
        state = rg.get_state(12345)
        assert isinstance(state, ChannelState)
        assert isinstance(state.lock, asyncio.Lock)
        assert state.messages == []

    def test_reuses_existing_state(self) -> None:
        """Возвращает тот же ChannelState при повторном вызове."""
        bot = MagicMock()
        rg = ReportGenerator(bot=bot)
        state1 = rg.get_state(12345)
        state2 = rg.get_state(12345)
        assert state1 is state2

    def test_different_channels_different_states(self) -> None:
        """Разные каналы — разные объекты состояния."""
        bot = MagicMock()
        rg = ReportGenerator(bot=bot)
        state1 = rg.get_state(111)
        state2 = rg.get_state(222)
        assert state1 is not state2


class TestReportGeneratorInit:
    """Тесты инициализации ReportGenerator."""

    def test_initial_state(self) -> None:
        """Начальное состояние — пустой словарь каналов."""
        bot = MagicMock()
        rg = ReportGenerator(bot=bot)
        assert rg.bot is bot
        assert rg.channels == {}


class TestReportGeneratorAddMessage:
    """Тесты метода add_message."""

    @pytest.mark.asyncio
    @patch("app.services.daily_report.get_channel_messages", new_callable=AsyncMock, return_value=[])
    @patch("app.services.daily_report.save_channel_message", new_callable=AsyncMock)
    async def test_initializes_channel_state(
        self, mock_save: AsyncMock, mock_get: AsyncMock
    ) -> None:
        """Первое сообщение инициализирует состояние канала."""
        bot = MagicMock()
        bot.report_msg_limit = 15
        rg = ReportGenerator(bot=bot)
        await rg.add_message(100, "привет", "user1", 1)

        assert 100 in rg.channels
        state = rg.channels[100]
        assert len(state.messages) == 1
        assert state.messages[0]["content"] == "привет"
        assert state.messages[0]["author"] == "user1"

    @pytest.mark.asyncio
    @patch("app.services.daily_report.get_channel_messages", new_callable=AsyncMock, return_value=[])
    @patch("app.services.daily_report.save_channel_message", new_callable=AsyncMock)
    async def test_accumulates_messages(self, mock_save: AsyncMock, mock_get: AsyncMock) -> None:
        """Несколько сообщений накапливаются."""
        bot = MagicMock()
        bot.report_msg_limit = 15
        rg = ReportGenerator(bot=bot)
        await rg.add_message(100, "привет", "user1", 1)
        await rg.add_message(100, "мир", "user2", 2)

        assert len(rg.channels[100].messages) == 2

    @pytest.mark.asyncio
    @patch("app.services.daily_report.get_channel_messages", new_callable=AsyncMock, return_value=[])
    @patch("app.services.daily_report.save_channel_message", new_callable=AsyncMock)
    async def test_calls_save_channel_message(
        self, mock_save: AsyncMock, mock_get: AsyncMock
    ) -> None:
        """Вызывает save_channel_message для сохранения в БД."""
        bot = MagicMock()
        bot.report_msg_limit = 15
        rg = ReportGenerator(bot=bot)
        await rg.add_message(100, "привет", "user1", 42)

        mock_save.assert_called_once_with(100, 42, "user1", "привет")

    @pytest.mark.asyncio
    async def test_handles_save_error_and_keeps_message_in_cache(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ошибка сохранения в БД не прерывает добавление сообщения в кэш."""
        bot = SimpleNamespace(report_msg_limit=15)
        rg = ReportGenerator(bot=bot)
        save_message = AsyncMock(side_effect=RuntimeError("database is unavailable"))
        get_messages = AsyncMock(return_value=[])
        monkeypatch.setattr(daily_report, "save_channel_message", save_message)
        monkeypatch.setattr(daily_report, "get_channel_messages", get_messages)

        await rg.add_message(100, "привет", "user1", 42)

        assert rg.channels[100].messages[0]["content"] == "привет"
        assert "Ошибка при сохранении сообщения в канал 100" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_handles_get_messages_error_and_uses_empty_count(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ошибка чтения БД обрабатывается как пустой список сообщений."""
        bot = SimpleNamespace(report_msg_limit=15)
        rg = ReportGenerator(bot=bot)
        save_message = AsyncMock()
        get_messages = AsyncMock(side_effect=RuntimeError("database is unavailable"))
        monkeypatch.setattr(daily_report, "save_channel_message", save_message)
        monkeypatch.setattr(daily_report, "get_channel_messages", get_messages)

        await rg.add_message(100, "привет", "user1", 42)

        save_message.assert_awaited_once_with(100, 42, "user1", "привет")
        assert rg.channels[100].messages
        assert "Ошибка при получении сообщений из канала 100" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_cancels_running_timer_when_activity_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Активный таймер отменяется при новом сообщении."""
        bot = SimpleNamespace(report_msg_limit=15)
        rg = ReportGenerator(bot=bot)
        state = rg.get_state(100)
        state.timer = MagicMock()
        state.timer.done.return_value = False
        monkeypatch.setattr(daily_report, "save_channel_message", AsyncMock())
        monkeypatch.setattr(daily_report, "get_channel_messages", AsyncMock(return_value=[]))

        await rg.add_message(100, "привет", "user1", 42)

        state.timer.cancel.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_does_not_cancel_completed_timer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Завершённый таймер не отменяется."""
        bot = SimpleNamespace(report_msg_limit=15)
        rg = ReportGenerator(bot=bot)
        state = rg.get_state(100)
        state.timer = MagicMock()
        state.timer.done.return_value = True
        monkeypatch.setattr(daily_report, "save_channel_message", AsyncMock())
        monkeypatch.setattr(daily_report, "get_channel_messages", AsyncMock(return_value=[]))

        await rg.add_message(100, "привет", "user1", 42)

        state.timer.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_starts_timer_when_cache_reaches_message_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """При достижении лимита по кэшу создаётся таймер отчёта."""
        bot = SimpleNamespace(report_msg_limit=1)
        rg = ReportGenerator(bot=bot)
        task = MagicMock()

        def create_task(coroutine: Any) -> MagicMock:
            coroutine.close()
            return task

        create_task_mock = MagicMock(side_effect=create_task)
        monkeypatch.setattr(daily_report.asyncio, "create_task", create_task_mock)
        monkeypatch.setattr(daily_report, "save_channel_message", AsyncMock())
        monkeypatch.setattr(daily_report, "get_channel_messages", AsyncMock(return_value=[]))

        await rg.add_message(100, "привет", "user1", 42)

        create_task_mock.assert_called_once()
        assert rg.channels[100].timer is task

    @pytest.mark.asyncio
    async def test_starts_timer_when_database_reaches_message_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """При достижении лимита в БД создаётся таймер даже при меньшем кэше."""
        bot = SimpleNamespace(report_msg_limit=3)
        rg = ReportGenerator(bot=bot)
        rg.get_state(100).messages.append({"id": 1})
        task = MagicMock()

        def create_task(coroutine: Any) -> MagicMock:
            coroutine.close()
            return task

        create_task_mock = MagicMock(side_effect=create_task)
        monkeypatch.setattr(daily_report.asyncio, "create_task", create_task_mock)
        monkeypatch.setattr(daily_report, "save_channel_message", AsyncMock())
        monkeypatch.setattr(
            daily_report,
            "get_channel_messages",
            AsyncMock(return_value=[_message(1), _message(2), _message(3)]),
        )

        await rg.add_message(100, "привет", "user1", 42)

        create_task_mock.assert_called_once()
        assert rg.channels[100].timer is task


class TestReportGeneratorTimer:
    """Тесты таймера ожидания перед генерацией отчёта."""

    @pytest.mark.asyncio
    async def test_returns_when_channel_state_was_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """После ожидания таймер завершается, если состояние канала удалено."""
        bot = SimpleNamespace(report_time_limit=5)
        rg = ReportGenerator(bot=bot)
        sleep = AsyncMock()
        generate = AsyncMock()
        monkeypatch.setattr(daily_report.asyncio, "sleep", sleep)
        monkeypatch.setattr(rg, "generate_and_send_report", generate)

        await rg.start_report_timer(100)

        sleep.assert_awaited_once_with(300)
        generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_when_channel_was_active_recently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Таймер не генерирует отчёт после недавней активности."""
        bot = SimpleNamespace(report_time_limit=5)
        rg = ReportGenerator(bot=bot)
        state = rg.get_state(100)
        state.last_message_time = datetime.now(tz=pytz.utc)
        sleep = AsyncMock()
        generate = AsyncMock()
        monkeypatch.setattr(daily_report.asyncio, "sleep", sleep)
        monkeypatch.setattr(rg, "generate_and_send_report", generate)

        await rg.start_report_timer(100)

        generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generates_report_after_inactivity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """После достаточного периода бездействия запускается генерация отчёта."""
        bot = SimpleNamespace(report_time_limit=1)
        rg = ReportGenerator(bot=bot)
        state = rg.get_state(100)
        state.last_message_time = datetime.now(tz=pytz.utc) - timedelta(minutes=2)
        monkeypatch.setattr(daily_report.asyncio, "sleep", AsyncMock())
        generate = AsyncMock()
        monkeypatch.setattr(rg, "generate_and_send_report", generate)

        await rg.start_report_timer(100)

        generate.assert_awaited_once_with(100)


class TestReportGeneratorGenerateAndSendReport:
    """Тесты генерации, отправки и очистки отчёта."""

    @pytest.mark.asyncio
    async def test_removes_state_when_channel_is_unavailable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Недоступный Discord-канал удаляется из состояния генератора."""
        bot = MagicMock()
        bot.get_channel.return_value = None
        rg = ReportGenerator(bot=bot)
        rg.get_state(100)

        await rg.generate_and_send_report(100)

        assert 100 not in rg.channels
        assert "Канал 100 недоступен" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_handles_error_loading_messages(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ошибка чтения сообщений останавливает генерацию без отправки."""
        bot = MagicMock()
        bot.report_msg_limit = 2
        bot.get_channel.return_value = MagicMock()
        rg = ReportGenerator(bot=bot)
        get_messages = AsyncMock(side_effect=RuntimeError("database is unavailable"))
        monkeypatch.setattr(daily_report, "get_channel_messages", get_messages)

        await rg.generate_and_send_report(100)

        assert (
            "Ошибка при получении сообщений для генерации отчета в канале 100"
            in capsys.readouterr().out
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("messages", "message_limit"),
        [([], 1), ([_message(1)], 2)],
    )
    async def test_returns_for_empty_or_insufficient_messages(
        self,
        messages: list[SimpleNamespace],
        message_limit: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Отчёт не создаётся при пустой или короткой истории сообщений."""
        bot = MagicMock()
        bot.report_msg_limit = message_limit
        bot.get_channel.return_value = MagicMock()
        rg = ReportGenerator(bot=bot)
        get_messages = AsyncMock(return_value=messages)
        get_client = MagicMock()
        monkeypatch.setattr(daily_report, "get_channel_messages", get_messages)
        monkeypatch.setattr(daily_report, "get_client", get_client)

        await rg.generate_and_send_report(100)

        get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_report_replaces_message_ids_and_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AI-отчёт отправляется со ссылками на сообщения, затем история очищается."""
        bot = MagicMock()
        bot.report_msg_limit = 2
        channel = MagicMock()
        channel.guild = SimpleNamespace(id=555)
        channel.send = AsyncMock()
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        rg.get_state(100)
        messages = [_message(1, "alice", "first"), _message(2, "bob", "second")]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Итог [ID:1]"))]
        )
        create = AsyncMock(return_value=response)
        client = MagicMock()
        client.chat.completions.create = create
        get_messages = AsyncMock(return_value=messages)
        delete_messages = AsyncMock()
        get_client = MagicMock(return_value=client)
        get_mini_model = MagicMock(return_value="mini-model")
        monkeypatch.setattr(daily_report, "get_channel_messages", get_messages)
        monkeypatch.setattr(daily_report, "delete_channel_messages", delete_messages)
        monkeypatch.setattr(daily_report, "get_client", get_client)
        monkeypatch.setattr(daily_report, "get_mini_model", get_mini_model)

        await rg.generate_and_send_report(100)

        expected_link = "https://discord.com/channels/555/100/1"
        channel.send.assert_awaited_once_with(f"Итог [ссылка]({expected_link})")
        delete_messages.assert_awaited_once_with(100)
        assert 100 not in rg.channels
        assert create.call_args.kwargs["model"] == "mini-model"
        assert "[ID:1] alice: first" in create.call_args.kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_uses_unknown_guild_for_channel_without_guild(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Для канала без guild ссылки получают идентификатор UNKNOWN."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = SimpleNamespace(send=AsyncMock())
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        messages = [_message(10)]
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Сводка [ID:10]"))]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        monkeypatch.setattr(daily_report, "get_channel_messages", AsyncMock(return_value=messages))
        monkeypatch.setattr(daily_report, "delete_channel_messages", AsyncMock())
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        channel.send.assert_awaited_once_with(
            "Сводка [ссылка](https://discord.com/channels/UNKNOWN/100/10)"
        )

    @pytest.mark.asyncio
    async def test_uses_fallback_for_empty_ai_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Пустой ответ AI превращается в диагностическое сообщение."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = MagicMock()
        channel.send = AsyncMock()
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        monkeypatch.setattr(
            daily_report, "get_channel_messages", AsyncMock(return_value=[_message(1)])
        )
        monkeypatch.setattr(daily_report, "delete_channel_messages", AsyncMock())
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        channel.send.assert_awaited_once_with("Пустой ответ от AI")

    @pytest.mark.asyncio
    async def test_handles_ai_generation_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ошибка AI не приводит к отправке или удалению сообщений."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = MagicMock()
        channel.send = AsyncMock()
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        create = AsyncMock(side_effect=RuntimeError("AI is unavailable"))
        client = MagicMock()
        client.chat.completions.create = create
        delete_messages = AsyncMock()
        monkeypatch.setattr(
            daily_report, "get_channel_messages", AsyncMock(return_value=[_message(1)])
        )
        monkeypatch.setattr(daily_report, "delete_channel_messages", delete_messages)
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        channel.send.assert_not_awaited()
        delete_messages.assert_not_awaited()
        assert "Ошибка генерации отчета" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_handles_send_error_without_deleting_messages(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ошибка отправки отчёта оставляет сообщения для повторной обработки."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = MagicMock()
        channel.send = AsyncMock(side_effect=RuntimeError("Discord is unavailable"))
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        delete_messages = AsyncMock()
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Сводка"))]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        monkeypatch.setattr(
            daily_report, "get_channel_messages", AsyncMock(return_value=[_message(1)])
        )
        monkeypatch.setattr(daily_report, "delete_channel_messages", delete_messages)
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        delete_messages.assert_not_awaited()
        assert 100 in rg.channels
        assert "Ошибка отправки отчета в канал 100" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_handles_delete_error_and_removes_state(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ошибка очистки БД логируется, а состояние канала всё равно удаляется."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = MagicMock()
        channel.send = AsyncMock()
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        rg.get_state(100)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Сводка"))]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        delete_messages = AsyncMock(side_effect=RuntimeError("database is unavailable"))
        monkeypatch.setattr(
            daily_report, "get_channel_messages", AsyncMock(return_value=[_message(1)])
        )
        monkeypatch.setattr(daily_report, "delete_channel_messages", delete_messages)
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        delete_messages.assert_awaited_once_with(100)
        assert 100 not in rg.channels
        assert "Ошибка при удалении сообщений из канала 100" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_skips_send_when_channel_becomes_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Защищённая ветка не отправляет отчёт, если канал стал недоступен."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = MagicMock()
        channel.__bool__.side_effect = [True, False]
        channel.send = AsyncMock()
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Сводка"))]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        monkeypatch.setattr(
            daily_report, "get_channel_messages", AsyncMock(return_value=[_message(1)])
        )
        monkeypatch.setattr(daily_report, "delete_channel_messages", AsyncMock())
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_succeeds_without_existing_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """После успешной отправки отсутствие состояния не вызывает ошибки очистки."""
        bot = MagicMock()
        bot.report_msg_limit = 1
        channel = MagicMock()
        channel.send = AsyncMock()
        bot.get_channel.return_value = channel
        rg = ReportGenerator(bot=bot)

        async def remove_state(_: str) -> None:
            rg.channels.pop(100, None)

        channel.send.side_effect = remove_state
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Сводка"))]
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        monkeypatch.setattr(
            daily_report, "get_channel_messages", AsyncMock(return_value=[_message(1)])
        )
        monkeypatch.setattr(daily_report, "delete_channel_messages", AsyncMock())
        monkeypatch.setattr(daily_report, "get_client", MagicMock(return_value=client))
        monkeypatch.setattr(daily_report, "get_mini_model", MagicMock(return_value="mini-model"))

        await rg.generate_and_send_report(100)

        channel.send.assert_awaited_once_with("Сводка")
        assert rg.channels == {}
