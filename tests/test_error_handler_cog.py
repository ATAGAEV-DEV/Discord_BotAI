"""Изолированные тесты обработчика ошибок без Discord, AI и реального ожидания."""

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call

import discord
import pytest
from discord.ext import commands

from app.cogs import error_handler
from app.core.bot import DisBot


@pytest.fixture(autouse=True)
def ai_generate(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Подменяет ссылку на handlers в Cog, не изменяя настоящий модуль обработчиков."""
    generate = AsyncMock(return_value="Ответ AI.")
    monkeypatch.setattr(error_handler, "handlers", SimpleNamespace(ai_generate=generate))
    return generate


@pytest.fixture(autouse=True)
def monotonic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Управляет временем только внутри Cog, не затрагивая часы event loop."""
    clock = MagicMock(return_value=100.0)
    monkeypatch.setattr(error_handler, "time", SimpleNamespace(monotonic=clock))
    return clock


@pytest.fixture
def mock_bot() -> MagicMock:
    """Создаёт бот с разными кэшами эмодзи серверов и асинхронной регистрацией Cog."""
    bot = MagicMock(spec=DisBot)
    bot.add_cog = AsyncMock()
    bot.command_prefix = "!"
    bot.context_limit = 17
    bot.guild_emoji_ids = {
        67890: {"smile": "111", "wave": "222"},
        98765: {"other": "333"},
        22222: {},
    }
    return bot


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Создаёт контекст с явными async-mock отправки и менеджера typing."""
    ctx = MagicMock(spec=commands.Context)
    ctx.author = MagicMock(spec=discord.Member)
    ctx.author.id = 12345
    ctx.author.mention = "<@12345>"
    ctx.guild = MagicMock(spec=discord.Guild)
    ctx.guild.id = 67890
    ctx.message = MagicMock(spec=discord.Message)
    ctx.message.content = "!Вопрос с пробелами?"
    ctx.command = MagicMock(spec=commands.Command)
    ctx.command.name = "search"
    ctx.command.signature = "<query>"
    ctx.send = AsyncMock()
    ctx.typing.return_value.__aenter__ = AsyncMock(return_value=None)
    ctx.typing.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


@pytest.fixture
def error_cog(mock_bot: MagicMock) -> error_handler.ErrorHandler:
    """Создаёт реальный Cog с независимым состоянием cooldown для каждого теста."""
    return error_handler.ErrorHandler(mock_bot)


@pytest.fixture
def non_ai_cog(error_cog: error_handler.ErrorHandler) -> error_handler.ErrorHandler:
    """Заполняет cooldown, чтобы обнаружить его случайное изменение при других ошибках."""
    error_cog._ai_cooldowns.update({12345: 99.0, 54321: 80.0})
    return error_cog


def assert_non_ai_response(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    expected: str,
) -> None:
    """Проверяет точный ответ и отсутствие побочных действий ветки AI."""
    mock_ctx.send.assert_called_once_with(expected)
    mock_ctx.send.assert_awaited_once_with(expected)
    ai_generate.assert_not_called()
    mock_ctx.typing.assert_not_called()
    monotonic.assert_not_called()
    assert error_cog._ai_cooldowns == {12345: 99.0, 54321: 80.0}
    assert error_cog.bot.mock_calls == []


def test_constructor_stores_bot_and_initializes_independent_cooldowns(
    mock_bot: MagicMock, ai_generate: AsyncMock, monotonic: MagicMock
) -> None:
    """Конструктор сохраняет бот и создаёт отдельный словарь с нулевым временем."""
    cog = error_handler.ErrorHandler(mock_bot)
    other = error_handler.ErrorHandler(mock_bot)

    assert cog.bot is mock_bot
    assert cog._ai_cooldowns == {}
    assert cog._ai_cooldowns[12345] == 0.0
    cog._ai_cooldowns[12345] = 50.0
    assert other._ai_cooldowns == {}
    assert other._ai_cooldowns is not cog._ai_cooldowns
    assert mock_bot.mock_calls == []
    ai_generate.assert_not_called()
    monotonic.assert_not_called()


@pytest.mark.asyncio
async def test_setup_registers_error_handler_once(
    mock_bot: MagicMock, ai_generate: AsyncMock, monotonic: MagicMock
) -> None:
    """Загрузка расширения ровно один раз ожидает регистрацию нужного Cog."""
    await error_handler.setup(mock_bot)

    mock_bot.add_cog.assert_awaited_once()
    cog = mock_bot.add_cog.await_args.args[0]
    assert isinstance(cog, error_handler.ErrorHandler)
    assert cog.bot is mock_bot
    assert cog._ai_cooldowns == {}
    assert mock_bot.mock_calls == [call.add_cog(cog)]
    ai_generate.assert_not_called()
    monotonic.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guild_id", "expected_emojis"),
    [
        (67890, {"smile": "111", "wave": "222"}),
        (98765, {"other": "333"}),
        (22222, {}),
        (11111, {}),
        (None, {}),
    ],
    ids=["first-guild", "second-guild", "empty-cache", "missing-cache", "private-message"],
)
async def test_unknown_command_generates_response_inside_typing(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    guild_id: int | None,
    expected_emojis: dict[str, str],
) -> None:
    """AI получает исходный текст, автора, лимит и эмодзи только текущего сервера."""
    if guild_id is None:
        mock_ctx.guild = None
    else:
        mock_ctx.guild.id = guild_id
    typing = mock_ctx.typing.return_value

    async def generate(*_args: object, **_kwargs: object) -> str:
        """Проверяет обновление cooldown и вход в typing до начала генерации."""
        assert error_cog._ai_cooldowns == {12345: 100.0}
        typing.__aenter__.assert_awaited_once_with()
        typing.__aexit__.assert_not_called()
        mock_ctx.send.assert_not_called()
        return "Ответ AI.\nВторая строка."

    async def send(_content: str) -> None:
        """Проверяет завершение генерации до отправки и выхода из typing."""
        ai_generate.assert_awaited_once()
        typing.__aexit__.assert_not_called()

    ai_generate.side_effect = generate
    mock_ctx.send.side_effect = send

    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    ai_generate.assert_awaited_once_with(
        "!Вопрос с пробелами?", guild_id, mock_ctx.author, limit=17, emoji_ids=expected_emojis
    )
    mock_ctx.send.assert_awaited_once_with("<@12345> Ответ AI.\nВторая строка.")
    mock_ctx.typing.assert_called_once_with()
    typing.__aenter__.assert_awaited_once_with()
    typing.__aexit__.assert_awaited_once_with(None, None, None)
    monotonic.assert_called_once_with()
    assert error_cog._ai_cooldowns == {12345: 100.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("elapsed", "remaining"), [(0.0, "5"), (1.4, "4"), (4.9, "0")]
)
async def test_ai_cooldown_blocks_without_updating_timestamp(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    elapsed: float,
    remaining: str,
) -> None:
    """Ранний повтор округляет остаток и не запускает AI или новый cooldown."""
    error_cog._ai_cooldowns.update({12345: 100.0, 54321: 80.0})
    monotonic.return_value = 100.0 + elapsed

    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    mock_ctx.send.assert_awaited_once_with(
        f"⏳ <@12345>, подождите {remaining} сек. перед следующим сообщением."
    )
    ai_generate.assert_not_called()
    mock_ctx.typing.assert_not_called()
    monotonic.assert_called_once_with()
    assert error_cog._ai_cooldowns == {12345: 100.0, 54321: 80.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [5.0, 5.1, 30.0], ids=["boundary", "just-after", "expired"])
async def test_ai_cooldown_allows_request_at_or_after_boundary(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    elapsed: float,
) -> None:
    """Ровно пять секунд и более разрешают запрос и обновляют время пользователя."""
    error_cog._ai_cooldowns[12345] = 100.0
    monotonic.return_value = 100.0 + elapsed

    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    ai_generate.assert_awaited_once()
    mock_ctx.send.assert_awaited_once_with("<@12345> Ответ AI.")
    mock_ctx.typing.assert_called_once_with()
    monotonic.assert_called_once_with()
    assert error_cog._ai_cooldowns == {12345: 100.0 + elapsed}


@pytest.mark.asyncio
async def test_blocked_attempt_does_not_extend_cooldown(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
) -> None:
    """После отклонённого повтора запрос разрешён через пять секунд от первого."""
    monotonic.side_effect = [100.0, 101.0, 105.0]

    for _ in range(3):
        await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    assert ai_generate.await_count == 2
    assert mock_ctx.send.await_args_list == [
        call("<@12345> Ответ AI."),
        call("⏳ <@12345>, подождите 4 сек. перед следующим сообщением."),
        call("<@12345> Ответ AI."),
    ]
    assert mock_ctx.typing.call_count == 2
    assert mock_ctx.typing.return_value.__aenter__.await_count == 2
    assert mock_ctx.typing.return_value.__aexit__.await_count == 2
    assert monotonic.call_count == 3
    assert error_cog._ai_cooldowns == {12345: 105.0}


@pytest.mark.asyncio
async def test_ai_cooldowns_are_independent_for_users(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
) -> None:
    """Разные пользователи могут обращаться к AI в один момент времени."""
    first_author = mock_ctx.author
    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    second_author = MagicMock(spec=discord.Member)
    second_author.id = 54321
    second_author.mention = "<@54321>"
    mock_ctx.author = second_author
    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    assert ai_generate.await_count == 2
    assert ai_generate.await_args_list[0].args[2] is first_author
    assert ai_generate.await_args_list[1].args[2] is second_author
    assert mock_ctx.send.await_args_list == [call("<@12345> Ответ AI."), call("<@54321> Ответ AI.")]
    assert monotonic.call_count == 2
    assert error_cog._ai_cooldowns == {12345: 100.0, 54321: 100.0}


@pytest.mark.asyncio
async def test_ai_cooldown_follows_user_across_guilds(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
) -> None:
    """Переход на другой сервер не сбрасывает ограничение того же пользователя."""
    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())
    mock_ctx.guild.id = 98765

    await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    ai_generate.assert_awaited_once()
    assert ai_generate.await_args.args[1] == 67890
    assert mock_ctx.send.await_args_list == [
        call("<@12345> Ответ AI."),
        call("⏳ <@12345>, подождите 5 сек. перед следующим сообщением."),
    ]
    mock_ctx.typing.assert_called_once_with()
    assert error_cog._ai_cooldowns == {12345: 100.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("generation failed"), TimeoutError("generation timeout")]
)
async def test_ai_failure_exits_typing_and_propagates(
    error_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    failure: Exception,
) -> None:
    """Ошибка внутри AI выходит из typing и пробрасывается без ложного ответа."""
    ai_generate.side_effect = failure

    with pytest.raises(type(failure)) as caught:
        await error_cog.on_command_error(mock_ctx, commands.CommandNotFound())

    assert caught.value is failure
    ai_generate.assert_awaited_once()
    mock_ctx.send.assert_not_called()
    mock_ctx.typing.return_value.__aenter__.assert_awaited_once_with()
    mock_ctx.typing.return_value.__aexit__.assert_awaited_once_with(type(failure), failure, ANY)
    assert error_cog._ai_cooldowns == {12345: 100.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            commands.MissingPermissions(["administrator"]),
            "❌ У вас недостаточно прав для выполнения этой команды.",
        ),
        (commands.NoPrivateMessage(), "❌ Эта команда недоступна в личных сообщениях."),
    ],
    ids=["missing-permissions", "no-private-message"],
)
async def test_permission_errors_send_exact_response(
    non_ai_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    capsys: pytest.CaptureFixture[str],
    error: commands.CommandError,
    expected: str,
) -> None:
    """Ошибки прав и личных сообщений отправляют только соответствующее пояснение."""
    if isinstance(error, commands.NoPrivateMessage):
        mock_ctx.guild = None

    await non_ai_cog.on_command_error(mock_ctx, error)

    assert_non_ai_response(non_ai_cog, mock_ctx, ai_generate, monotonic, expected)
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_after", "rounded"), [(0.0, "0"), (1.4, "1"), (1.6, "2"), (2.5, "2")]
)
async def test_command_cooldown_rounds_retry_after(
    non_ai_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    capsys: pytest.CaptureFixture[str],
    retry_after: float,
    rounded: str,
) -> None:
    """Cooldown команды округляет время до целого, не затрагивая cooldown AI."""
    error = commands.CommandOnCooldown(
        commands.Cooldown(1, 10.0), retry_after, commands.BucketType.user
    )

    await non_ai_cog.on_command_error(mock_ctx, error)

    assert_non_ai_response(
        non_ai_cog,
        mock_ctx,
        ai_generate,
        monotonic,
        f"⏳ Подождите {rounded} сек. перед повторным использованием.",
    )
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefix", "name", "signature"),
    [("!", "search", "<query>"), ("??", "configure", "<key> [value=default]")],
)
async def test_missing_argument_includes_prefix_name_and_signature(
    non_ai_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    capsys: pytest.CaptureFixture[str],
    prefix: str,
    name: str,
    signature: str,
) -> None:
    """Подсказка использования учитывает текущие префикс, имя и сигнатуру команды."""
    non_ai_cog.bot.command_prefix = prefix
    mock_ctx.command.name = name
    mock_ctx.command.signature = signature
    parameter = commands.Parameter("query", commands.Parameter.POSITIONAL_OR_KEYWORD)

    await non_ai_cog.on_command_error(mock_ctx, commands.MissingRequiredArgument(parameter))

    assert_non_ai_response(
        non_ai_cog,
        mock_ctx,
        ai_generate,
        monotonic,
        f"❌ Неправильное использование команды. Используйте: `{prefix}{name} {signature}`",
    )
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "command-invoke-error"])
async def test_network_errors_are_recognized_directly_and_wrapped(
    non_ai_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
    wrapped: bool,
) -> None:
    """Сетевая ошибка определяется как напрямую, так и через поле original."""
    original = error_type("network failure")
    error = commands.CommandInvokeError(original) if wrapped else original

    await non_ai_cog.on_command_error(mock_ctx, error)

    assert_non_ai_response(
        non_ai_cog,
        mock_ctx,
        ai_generate,
        monotonic,
        "❌ Проблема с сетью. Попробуйте позже.",
    )
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        commands.CommandError("plain failure"),
        commands.BadArgument("bad input"),
        commands.CommandInvokeError(ValueError("invalid result")),
        commands.CommandInvokeError(commands.MissingPermissions(["administrator"])),
    ],
    ids=["generic", "bad-argument", "wrapped-value-error", "wrapped-permission-error"],
)
async def test_unhandled_error_sends_generic_response_and_prints_outer_error(
    non_ai_cog: error_handler.ErrorHandler,
    mock_ctx: MagicMock,
    ai_generate: AsyncMock,
    monotonic: MagicMock,
    capsys: pytest.CaptureFixture[str],
    error: commands.CommandError,
) -> None:
    """Неизвестная ошибка печатается целиком; original разворачивается лишь для сети."""
    await non_ai_cog.on_command_error(mock_ctx, error)

    assert_non_ai_response(
        non_ai_cog,
        mock_ctx,
        ai_generate,
        monotonic,
        "❌ Произошла ошибка при выполнении команды.",
    )
    assert capsys.readouterr().out == f"Command error: {error}\n"
