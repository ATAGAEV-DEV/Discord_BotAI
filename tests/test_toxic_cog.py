"""Поведенческие тесты прожарки и штатного cooldown без внешних запросов."""

import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call

import discord
import pytest
from discord.ext import commands
from discord.ext.commands.view import StringView

from app.cogs import toxic
from app.core.bot import DisBot


def _message(
    content: str,
    name: str = "Alice",
    *,
    author: SimpleNamespace | None = None,
    attachments: bool = False,
    stickers: bool = False,
) -> SimpleNamespace:
    """Создаёт сообщение истории с явно заданными текстом и вложениями."""
    return SimpleNamespace(
        content=content,
        author=author if author is not None else SimpleNamespace(name=name, bot=False),
        attachments=[object()] if attachments else [],
        stickers=[object()] if stickers else [],
    )


@pytest.fixture
def toxic_env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Изолирует историю, AI и кэш, оставляя настоящую обработку команды."""
    bot = MagicMock(spec=DisBot)
    bot.user = SimpleNamespace(id=999, name="RoastBot", bot=True)
    bot.add_cog = AsyncMock()
    ctx = MagicMock(spec=commands.Context)
    ctx.prefix = "!"
    ctx.send = AsyncMock()
    ctx.channel = SimpleNamespace()
    ctx.guild = SimpleNamespace(id=456)
    ctx.typing = MagicMock()
    typing = ctx.typing.return_value
    typing.__aenter__ = AsyncMock(return_value=None)
    typing.__aexit__ = AsyncMock(return_value=False)
    messages = [_message("Привет")]

    async def history(*, limit: int) -> AsyncIterator[SimpleNamespace]:
        """Возвращает не более запрошенного числа сообщений, как Discord history."""
        for message in messages[:limit]:
            yield message

    ctx.channel.history = MagicMock(side_effect=history)
    response = SimpleNamespace(content="Готовая прожарка")
    create = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=response)])
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    get_client = MagicMock(return_value=client)
    get_model = MagicMock(return_value="test-roast-model")
    get_descriptions = MagicMock(return_value={"Alice": "любит Python", "Bob": "играет"})
    monkeypatch.setattr(toxic, "get_client", get_client)
    monkeypatch.setattr(toxic, "get_model", get_model)
    monkeypatch.setattr(
        toxic, "user_descriptions_cache", SimpleNamespace(get_all=get_descriptions)
    )
    monkeypatch.setattr(toxic, "ROAST_PROMPT", "Системная роль.\n{user_info}")
    monkeypatch.setattr(toxic, "ROAST_PERSONAS", {"babka": "Роль бабки", "robot": "Роль робота"})
    return SimpleNamespace(
        bot=bot,
        cog=toxic.Toxic(bot),
        ctx=ctx,
        typing=typing,
        messages=messages,
        response=response,
        create=create,
        get_client=get_client,
        get_model=get_model,
        get_descriptions=get_descriptions,
    )


def _assert_no_ai(env: SimpleNamespace) -> None:
    """Проверяет отсутствие AI-запроса и даже входа в typing-контекст."""
    env.get_client.assert_not_called()
    env.get_model.assert_not_called()
    env.create.assert_not_called()
    env.ctx.typing.assert_not_called()
    env.typing.__aenter__.assert_not_called()
    env.typing.__aexit__.assert_not_called()


def _assert_successful_roast(
    env: SimpleNamespace,
    *,
    history_limit: int = 40,
    history_text: str = "[Alice]: Привет",
    system_content: str = "Системная роль.\n- Alice: любит Python\n- Bob: играет",
    response: str = "Готовая прожарка",
) -> None:
    """Проверяет полный AI-запрос, ожидание отправки и закрытие typing-контекста."""
    env.ctx.channel.history.assert_called_once_with(limit=history_limit)
    env.get_descriptions.assert_called_once_with()
    env.get_client.assert_called_once_with()
    env.get_model.assert_called_once_with()
    expected_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Вот последние сообщения чата:\n{history_text}"},
    ]
    env.create.assert_called_once_with(
        model="test-roast-model", messages=expected_messages, temperature=0.9, max_tokens=3000
    )
    env.create.assert_awaited_once_with(
        model="test-roast-model", messages=expected_messages, temperature=0.9, max_tokens=3000
    )
    env.ctx.typing.assert_called_once_with()
    env.typing.__aenter__.assert_awaited_once_with()
    env.typing.__aexit__.assert_awaited_once_with(None, None, None)
    env.ctx.send.assert_called_once_with(response)
    env.ctx.send.assert_awaited_once_with(response)


def test_constructor_keeps_bot_without_external_calls(toxic_env: SimpleNamespace) -> None:
    """Создание Cog не запускает AI, регистрацию или чтение истории."""
    assert toxic_env.cog.bot is toxic_env.bot
    assert toxic_env.cog.roast_command.name == "toxic"
    toxic_env.bot.add_cog.assert_not_called()
    toxic_env.ctx.channel.history.assert_not_called()
    toxic_env.get_descriptions.assert_not_called()
    _assert_no_ai(toxic_env)


@pytest.mark.asyncio
async def test_setup_registers_toxic_once(toxic_env: SimpleNamespace) -> None:
    """Setup ожидает регистрацию правильного Cog с исходным ботом."""
    await toxic.setup(toxic_env.bot)

    toxic_env.bot.add_cog.assert_awaited_once()
    cog = toxic_env.bot.add_cog.await_args.args[0]
    assert isinstance(cog, toxic.Toxic)
    assert cog.bot is toxic_env.bot
    assert toxic_env.bot.mock_calls == [call.add_cog(cog)]


@pytest.mark.asyncio
async def test_setup_propagates_registration_error(toxic_env: SimpleNamespace) -> None:
    """Ошибка регистрации распространяется без повторной попытки."""
    error = RuntimeError("registration failed")
    toxic_env.bot.add_cog.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await toxic.setup(toxic_env.bot)

    assert exc_info.value is error
    toxic_env.bot.add_cog.assert_awaited_once()
    _assert_no_ai(toxic_env)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "limit"),
    [
        ((), 20),
        (("0",), 1),
        (("1",), 1),
        (("20",), 20),
        (("80",), 80),
        (("81",), 80),
        (("999999",), 80),
        (("5", "2"), 2),
    ],
    ids=["default", "zero", "min", "normal", "max", "over-max", "large", "last-number"],
)
async def test_numeric_arguments_bound_both_history_and_ai_context(
    toxic_env: SimpleNamespace, args: tuple[str, ...], limit: int
) -> None:
    """Границы 1..80 применяются и к запросу истории, и к числу сообщений для AI."""
    toxic_env.messages[:] = [_message(f"message-{i}") for i in range(82)]

    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx, *args)

    expected_history = "\n".join(f"[Alice]: message-{i}" for i in reversed(range(limit)))
    _assert_successful_roast(toxic_env, history_limit=limit * 2, history_text=expected_history)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "history_limit", "role"),
    [
        (("babka",), 40, "Роль бабки"),
        (("2", "babka"), 4, "Роль бабки"),
        (("babka", "2"), 4, "Роль бабки"),
        (("babka", "robot"), 40, "Роль робота"),
        (("missing", "babka"), 40, "Роль бабки"),
    ],
    ids=["persona-only", "number-first", "persona-first", "last-persona", "replace-invalid"],
)
async def test_persona_arguments_extend_system_prompt(
    toxic_env: SimpleNamespace, args: tuple[str, ...], history_limit: int, role: str
) -> None:
    """Порядок числа и роли не важен; последнее имя роли определяет дополнение."""
    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx, *args)

    _assert_successful_roast(
        toxic_env,
        history_limit=history_limit,
        system_content=(
            "Системная роль.\n- Alice: любит Python\n- Bob: играет"
            f"\n\nВАЖНОЕ ДОПОЛНЕНИЕ К РОЛИ:\n{role}"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [("list",), ("2", "list"), ("babka", "list")])
async def test_list_returns_personas_without_history_or_ai(
    toxic_env: SimpleNamespace, args: tuple[str, ...]
) -> None:
    """Список режимов доступен без чтения истории, кэша описаний и вызова AI."""
    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx, *args)

    toxic_env.ctx.send.assert_awaited_once_with("🎭 **Доступные режимы:** `babka`, `robot`")
    toxic_env.ctx.channel.history.assert_not_called()
    toxic_env.get_descriptions.assert_not_called()
    _assert_no_ai(toxic_env)


@pytest.mark.asyncio
@pytest.mark.parametrize("persona", ["missing", "-1", "BAbka"])
async def test_unknown_persona_reports_error_without_ai(
    toxic_env: SimpleNamespace, persona: str
) -> None:
    """Неизвестные имена, отрицательное число и другой регистр не принимаются как роль."""
    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx, persona)

    toxic_env.ctx.channel.history.assert_called_once_with(limit=40)
    toxic_env.ctx.send.assert_awaited_once_with(
        f"❌ Нет такого режима `{persona}`. Доступные: `babka`, `robot`"
    )
    _assert_no_ai(toxic_env)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["!", "~"])
async def test_history_filters_and_reverses_only_selected_messages(
    toxic_env: SimpleNamespace, prefix: str
) -> None:
    """Собственный бот, команды и пустые записи исключаются до хронологической сборки."""
    toxic_env.ctx.prefix = prefix
    toxic_env.messages[:] = [
        _message("Не включать", author=toxic_env.bot.user),
        _message(f"{prefix}help"),
        _message(""),
        _message("Последнее", "Bob"),
        _message(f"Префикс {prefix} внутри текста"),
        _message("Самое раннее", "Carol"),
    ]

    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    _assert_successful_roast(
        toxic_env,
        history_text=(
            f"[Carol]: Самое раннее\n[Alice]: Префикс {prefix} внутри текста\n[Bob]: Последнее"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "attachments", "stickers", "expected"),
    [
        ("", True, False, "[Пользователь скинул картинку/файл]"),
        ("", False, True, "[Пользователь отправил стикер]"),
        ("", True, True, "[Пользователь скинул картинку/файл]"),
        ("Подпись", True, True, "Подпись"),
        ("http://example.test", False, False, "[Пользователь отправил ссылку]"),
        ("https://example.test", False, False, "[Пользователь отправил ссылку]"),
        ("Смотри https://example.test", False, False, "Смотри https://example.test"),
    ],
    ids=["attachment", "sticker", "attachment-first", "caption", "http", "https", "inline-url"],
)
async def test_history_normalizes_media_and_links(
    toxic_env: SimpleNamespace, content: str, attachments: bool, stickers: bool, expected: str
) -> None:
    """Медиа без текста получают маркеры; подписи и ссылки внутри текста сохраняются."""
    toxic_env.messages[:] = [_message(content, attachments=attachments, stickers=stickers)]

    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    _assert_successful_roast(toxic_env, history_text=f"[Alice]: {expected}")


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True], ids=["empty", "all-filtered"])
async def test_empty_context_reports_silence_without_ai(
    toxic_env: SimpleNamespace, filtered: bool
) -> None:
    """И пустая история, и отфильтрованная история завершаются до запроса AI."""
    toxic_env.messages.clear()
    if filtered:
        toxic_env.messages.extend(
            [_message("Бот", author=toxic_env.bot.user), _message("!help"), _message("")]
        )

    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    toxic_env.ctx.channel.history.assert_called_once_with(limit=40)
    toxic_env.ctx.send.assert_awaited_once_with("Тут слишком тихо, некого прожаривать. 🦗")
    toxic_env.get_descriptions.assert_not_called()
    _assert_no_ai(toxic_env)


@pytest.mark.asyncio
async def test_empty_descriptions_still_build_valid_request(toxic_env: SimpleNamespace) -> None:
    """Отсутствие описаний не мешает отправить историю с системной ролью."""
    toxic_env.get_descriptions.return_value = {}

    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    _assert_successful_roast(toxic_env, system_content="Системная роль.\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "**Жирный** и *курсив*, __подчёркивание__, ~~ошибка~~",
            "Жирный и курсив, подчёркивание, ошибка",
        ),
        ("Текст без разметки", "Текст без разметки"),
        (None, ""),
        ("", ""),
    ],
    ids=["markdown", "plain", "none", "empty"],
)
async def test_response_is_cleaned_with_real_discord_utility(
    toxic_env: SimpleNamespace, content: str | None, expected: str
) -> None:
    """Markdown удаляется; пустой content лишь передаётся в send, без гарантии доставки."""
    toxic_env.response.content = content

    await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    _assert_successful_roast(toxic_env, response=expected)


@pytest.mark.asyncio
async def test_ai_error_exits_typing_without_sending(toxic_env: SimpleNamespace) -> None:
    """Сбой AI распространяется и закрывает typing без отправки и повторного запроса."""
    error = RuntimeError("AI failed")
    toxic_env.create.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    assert exc_info.value is error
    toxic_env.get_client.assert_called_once_with()
    toxic_env.get_model.assert_called_once_with()
    toxic_env.create.assert_awaited_once_with(
        model="test-roast-model",
        messages=[
            {"role": "system", "content": "Системная роль.\n- Alice: любит Python\n- Bob: играет"},
            {"role": "user", "content": "Вот последние сообщения чата:\n[Alice]: Привет"},
        ],
        temperature=0.9,
        max_tokens=3000,
    )
    toxic_env.typing.__aenter__.assert_awaited_once_with()
    toxic_env.typing.__aexit__.assert_awaited_once_with(RuntimeError, error, ANY)
    toxic_env.ctx.send.assert_not_called()


@pytest.mark.asyncio
async def test_history_error_stops_before_ai(toxic_env: SimpleNamespace) -> None:
    """Ошибка при итерации истории прерывает обработку даже после первого сообщения."""
    error = RuntimeError("history failed")

    async def failing_history(*, limit: int) -> AsyncIterator[SimpleNamespace]:
        """Возвращает часть истории и имитирует сбой следующей страницы Discord."""
        yield _message("Частичная история")
        raise error

    toxic_env.ctx.channel.history.side_effect = failing_history

    with pytest.raises(RuntimeError) as exc_info:
        await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    assert exc_info.value is error
    toxic_env.ctx.channel.history.assert_called_once_with(limit=40)
    toxic_env.get_descriptions.assert_not_called()
    toxic_env.ctx.send.assert_not_called()
    _assert_no_ai(toxic_env)


@pytest.mark.asyncio
async def test_send_error_exits_typing_without_retrying_ai(toxic_env: SimpleNamespace) -> None:
    """Ошибка отправки закрывает typing без повторного AI-запроса или отправки."""
    error = RuntimeError("send failed")
    toxic_env.ctx.send.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        await toxic_env.cog.roast_command.callback(toxic_env.cog, toxic_env.ctx)

    assert exc_info.value is error
    toxic_env.create.assert_awaited_once()
    toxic_env.ctx.send.assert_awaited_once_with("Готовая прожарка")
    toxic_env.typing.__aenter__.assert_awaited_once_with()
    toxic_env.typing.__aexit__.assert_awaited_once_with(RuntimeError, error, ANY)


@pytest.fixture
async def invoked_env(toxic_env: SimpleNamespace) -> AsyncIterator[SimpleNamespace]:
    """Регистрирует новый Cog в настоящем Bot с независимым состоянием команды."""
    async with commands.Bot(command_prefix="!", intents=discord.Intents.none()) as bot:
        cog = toxic.Toxic(bot)
        await bot.add_cog(cog)
        command = bot.get_command("toxic")
        assert command is cog.roast_command
        assert command is not toxic.Toxic.roast_command
        # Cog копирует Command, но discord.py сохраняет mapping из декоратора.
        # Изолируем лишь состояние, оставляя реальные параметры и алгоритм cooldown.
        cooldown = command._buckets._cooldown
        assert cooldown is not None
        command._buckets = commands.CooldownMapping.from_cooldown(
            cooldown.rate, cooldown.per, command._buckets.type
        )
        toxic_env.bot = bot
        toxic_env.cog = cog
        toxic_env.command = command
        yield toxic_env


def _invocation_context(
    env: SimpleNamespace,
    *,
    seconds: float = 0,
    user_id: int = 123,
    guild_id: int = 456,
    arguments: str = "",
) -> commands.Context:
    """Создаёт настоящий Context и StringView с фиксированным временем сообщения."""
    message = SimpleNamespace(
        _state=env.bot._connection,
        author=SimpleNamespace(id=user_id, bot=False),
        guild=SimpleNamespace(id=guild_id),
        channel=env.ctx.channel,
        attachments=[],
        edited_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
    )
    ctx = commands.Context(
        message=message,
        bot=env.bot,
        view=StringView(arguments),
        prefix="!",
        command=env.command,
        invoked_with="toxic",
    )
    ctx.send = env.ctx.send
    ctx.typing = env.ctx.typing
    return ctx


@pytest.mark.asyncio
async def test_real_command_parses_arguments_on_first_invocation(
    invoked_env: SimpleNamespace,
) -> None:
    """Первый вызов проходит проверки, штатный разбор аргументов и тело команды."""
    ctx = _invocation_context(invoked_env, arguments="2 babka")

    await invoked_env.command.invoke(ctx)

    assert ctx.args == [invoked_env.cog, ctx, "2", "babka"]
    assert ctx.command_failed is False
    _assert_successful_roast(
        invoked_env,
        history_limit=4,
        system_content=(
            "Системная роль.\n- Alice: любит Python\n- Bob: играет"
            "\n\nВАЖНОЕ ДОПОЛНЕНИЕ К РОЛИ:\nРоль бабки"
        ),
    )


@pytest.mark.asyncio
async def test_cooldown_rejects_repeat_before_thirty_seconds(
    invoked_env: SimpleNamespace,
) -> None:
    """Повтор до 30 секунд блокируется до чтения истории, typing и AI-запроса."""
    await invoked_env.command.invoke(_invocation_context(invoked_env))
    _assert_successful_roast(invoked_env)

    with pytest.raises(commands.CommandOnCooldown) as exc_info:
        await invoked_env.command.invoke(_invocation_context(invoked_env, seconds=5))

    assert exc_info.value.retry_after == pytest.approx(25.0)
    assert exc_info.value.type is commands.BucketType.user
    assert exc_info.value.cooldown.rate == 1
    assert exc_info.value.cooldown.per == 30.0
    _assert_successful_roast(invoked_env)


@pytest.mark.asyncio
async def test_cooldown_accepts_invocation_after_interval(invoked_env: SimpleNamespace) -> None:
    """Интервал проверяется по timestamps без реального ожидания 30 секунд."""
    await invoked_env.command.invoke(_invocation_context(invoked_env))
    await invoked_env.command.invoke(_invocation_context(invoked_env, seconds=30.001))

    assert invoked_env.ctx.channel.history.call_args_list == [call(limit=40), call(limit=40)]
    assert invoked_env.create.await_count == 2
    assert invoked_env.ctx.send.await_args_list == [
        call("Готовая прожарка"), call("Готовая прожарка")
    ]
    assert invoked_env.typing.__aenter__.await_count == 2
    assert invoked_env.typing.__aexit__.await_args_list == [
        call(None, None, None), call(None, None, None)
    ]


@pytest.mark.asyncio
async def test_cooldown_is_independent_for_different_users(invoked_env: SimpleNamespace) -> None:
    """Другой пользователь выполняет команду в том же временном окне."""
    await invoked_env.command.invoke(_invocation_context(invoked_env, user_id=123))
    await invoked_env.command.invoke(_invocation_context(invoked_env, seconds=5, user_id=124))

    assert invoked_env.ctx.channel.history.call_count == 2
    assert invoked_env.create.await_count == 2
    assert invoked_env.ctx.send.await_args_list == [
        call("Готовая прожарка"), call("Готовая прожарка")
    ]


@pytest.mark.asyncio
async def test_user_cooldown_is_shared_across_guilds(invoked_env: SimpleNamespace) -> None:
    """BucketType.user ограничивает пользователя и при переходе на другой сервер."""
    await invoked_env.command.invoke(_invocation_context(invoked_env, guild_id=456))

    with pytest.raises(commands.CommandOnCooldown) as exc_info:
        await invoked_env.command.invoke(
            _invocation_context(invoked_env, seconds=5, guild_id=789)
        )

    assert exc_info.value.retry_after == pytest.approx(25.0)
    assert exc_info.value.type is commands.BucketType.user
    _assert_successful_roast(invoked_env)


@pytest.mark.asyncio
async def test_real_bot_loads_both_extensions_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Настоящий загрузчик регистрирует Cogs, команду и слушатель без login к Discord."""
    # load_extension временно заменяет sys.modules; восстановление не мешает другим тестам.
    for extension in ("app.cogs.ranks", "app.cogs.toxic"):
        monkeypatch.setitem(sys.modules, extension, sys.modules[extension])
    async with commands.Bot(command_prefix="!", intents=discord.Intents.none()) as bot:
        await bot.load_extension("app.cogs.ranks")
        await bot.load_extension("app.cogs.toxic")
        rank_cog = bot.get_cog("Ranks")
        toxic_cog = bot.get_cog("Toxic")

        assert isinstance(rank_cog, bot.extensions["app.cogs.ranks"].Ranks)
        assert isinstance(toxic_cog, bot.extensions["app.cogs.toxic"].Toxic)
        assert rank_cog.bot is bot
        assert toxic_cog.bot is bot
        assert rank_cog.get_listeners() == [("on_message", rank_cog.on_message)]
        assert rank_cog.on_message in bot.extra_events["on_message"]
        assert bot.get_command("toxic") is toxic_cog.roast_command
        assert toxic_cog.roast_command.cog is toxic_cog

    assert bot.is_closed()
    assert bot.get_cog("Ranks") is None
    assert bot.get_cog("Toxic") is None
    assert bot.get_command("toxic") is None
    assert not bot.extra_events["on_message"]
