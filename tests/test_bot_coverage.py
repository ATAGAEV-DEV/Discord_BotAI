"""Изолированные тесты жизненного цикла бота и фильтрации сообщений."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import discord
import pytest

from app.core import bot as bot_module
from app.core.bot import DisBot

_EXTENSIONS = (
    "app.cogs.general",
    "app.cogs.admin",
    "app.cogs.youtube",
    "app.cogs.toxic",
    "app.cogs.nicknames",
    "app.cogs.error_handler",
    "app.cogs.ranks",
)
_READY_STEPS = (
    "init_models",
    "load_admins",
    "load_user_descriptions",
    "load_emoji_descriptions",
    "create_report_generator",
)


@pytest.fixture
def isolated_bot(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Создаёт бот без Discord-клиента, БД и фоновых задач."""
    bot = MagicMock(spec=DisBot)
    bot.command_prefix = "!"
    bot.report_generator = MagicMock(spec=bot_module.ReportGenerator)
    bot.report_generator.add_message = AsyncMock()
    bot.youtube_notifier = MagicMock(spec=bot_module.YouTubeNotifier)
    bot.load_extension = AsyncMock()
    bot.process_commands = AsyncMock()
    bot.guilds = []
    bot.guild_emoji_ids = {}
    bot._build_emoji_cache.side_effect = lambda guild: DisBot._build_emoji_cache(bot, guild)
    monkeypatch.setattr(bot_module, "REPORT_IGNORE_PREFIX", "?")
    return bot


@pytest.fixture
def ready_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Подменяет зависимости on_ready и позволяет проверять порядок вызовов."""
    dependencies = MagicMock()
    dependencies.init_models = AsyncMock()
    dependencies.load_admins = AsyncMock()
    dependencies.load_user_descriptions = AsyncMock()
    dependencies.load_emoji_descriptions = AsyncMock()
    dependencies.create_report_generator = MagicMock()
    dependencies.create_report_generator.return_value = MagicMock(spec=bot_module.ReportGenerator)
    monkeypatch.setattr(bot_module, "init_models", dependencies.init_models)
    monkeypatch.setattr(bot_module.admins, "load_all", dependencies.load_admins)
    monkeypatch.setattr(
        bot_module.user_descriptions_cache, "load_all", dependencies.load_user_descriptions
    )
    monkeypatch.setattr(
        bot_module.emoji_descriptions_cache, "load_all", dependencies.load_emoji_descriptions
    )
    monkeypatch.setattr(bot_module, "ReportGenerator", dependencies.create_report_generator)
    return dependencies


def _guild(guild_id: int, name: str, emojis: list[tuple[str, int]]) -> SimpleNamespace:
    """Создаёт гильдию с конкретными именами и числовыми ID эмодзи."""
    return SimpleNamespace(
        id=guild_id,
        name=name,
        emojis=[SimpleNamespace(name=emoji_name, id=emoji_id) for emoji_name, emoji_id in emojis],
    )


def _message(content: str, *, author_is_bot: bool = False) -> MagicMock:
    """Создаёт сообщение с асинхронной отправкой ответа в канал."""
    message = MagicMock(spec=discord.Message)
    message.author.bot = author_is_bot
    message.author.display_name = "Test User"
    message.content = content
    message.channel.id = 123
    message.channel.send = AsyncMock()
    message.id = 456
    return message


@pytest.mark.parametrize("limits", [None, (42, 4, 10)], ids=["defaults", "custom"])
def test_constructor_passes_discord_options_and_initializes_state(
    monkeypatch: pytest.MonkeyPatch, limits: tuple[int, int, int] | None
) -> None:
    """Конструктор передаёт настройки Discord и сохраняет лимиты и сервисы."""
    notifier_class = MagicMock(spec=bot_module.YouTubeNotifier)
    monkeypatch.setattr(bot_module, "YouTubeNotifier", notifier_class)
    intents = discord.Intents.none()
    help_command = None if limits is None else MagicMock(spec=bot_module.commands.HelpCommand)
    options = (
        {}
        if limits is None
        else {
            "context_limit": limits[0],
            "report_msg_limit": limits[1],
            "report_time_limit": limits[2],
        }
    )

    with patch.object(bot_module.commands.Bot, "__init__", autospec=True, return_value=None) as init:
        bot = DisBot(command_prefix="~", intents=intents, help_command=help_command, **options)

    init.assert_called_once_with(bot, command_prefix="~", intents=intents, help_command=help_command)
    notifier_class.assert_called_once_with(bot)
    assert bot.youtube_notifier is notifier_class.return_value
    assert bot.report_generator is None
    assert bot.guild_emoji_ids == {}
    expected_limits = limits or (
        bot_module.CONTEXT_LIMIT,
        bot_module.REPORT_MSG_LIMIT,
        bot_module.REPORT_TIME_LIMIT,
    )
    assert (bot.context_limit, bot.report_msg_limit, bot.report_time_limit) == expected_limits


def test_constructor_creates_independent_emoji_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Кэш эмодзи не разделяется между экземплярами бота."""
    monkeypatch.setattr(bot_module, "YouTubeNotifier", MagicMock())
    with patch.object(bot_module.commands.Bot, "__init__", autospec=True, return_value=None):
        first = DisBot(command_prefix="!", intents=discord.Intents.none())
        second = DisBot(command_prefix="!", intents=discord.Intents.none())

    first.guild_emoji_ids[100] = {"smile": "123"}

    assert second.guild_emoji_ids == {}
    assert first.guild_emoji_ids is not second.guild_emoji_ids


@pytest.mark.parametrize(
    ("emojis", "expected"),
    [
        ([], {}),
        (
            [("smile", 123), ("wave", 987654321012345678)],
            {"smile": "123", "wave": "987654321012345678"},
        ),
        ([("smile", 123), ("smile", 456)], {"smile": "456"}),
    ],
    ids=["empty", "multiple", "duplicate-names"],
)
def test_build_emoji_cache_uses_names_and_string_ids(
    isolated_bot: MagicMock, emojis: list[tuple[str, int]], expected: dict[str, str]
) -> None:
    """Кэш строится из эмодзи гильдии, не изменяя состояние бота."""
    guild = _guild(100, "Test Guild", emojis)

    result = DisBot._build_emoji_cache(isolated_bot, guild)

    assert result == expected
    assert isolated_bot.guild_emoji_ids == {}
    assert [(emoji.name, emoji.id) for emoji in guild.emojis] == emojis


@pytest.mark.asyncio
async def test_setup_hook_loads_extensions_before_starting_scheduler(
    isolated_bot: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Все расширения загружаются по порядку до запуска планировщика."""
    scheduler = MagicMock()
    monkeypatch.setattr(bot_module, "start_scheduler", scheduler)
    calls = MagicMock()
    calls.attach_mock(isolated_bot.load_extension, "load_extension")
    calls.attach_mock(scheduler, "start_scheduler")

    await DisBot.setup_hook(isolated_bot)

    assert isolated_bot.load_extension.await_args_list == [call(name) for name in _EXTENSIONS]
    scheduler.assert_called_once_with(isolated_bot, isolated_bot.youtube_notifier)
    assert calls.mock_calls == [call.load_extension(name) for name in _EXTENSIONS] + [
        call.start_scheduler(isolated_bot, isolated_bot.youtube_notifier)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_extension", _EXTENSIONS)
async def test_setup_hook_stops_after_extension_failure(
    isolated_bot: MagicMock, monkeypatch: pytest.MonkeyPatch, failed_extension: str
) -> None:
    """Ошибка загрузки распространяется без загрузки следующих Cogs и запуска задач."""
    scheduler = MagicMock()
    monkeypatch.setattr(bot_module, "start_scheduler", scheduler)
    failed_index = _EXTENSIONS.index(failed_extension)
    error = RuntimeError("extension failed")
    isolated_bot.load_extension.side_effect = [None] * failed_index + [error]

    with pytest.raises(RuntimeError, match="extension failed") as raised:
        await DisBot.setup_hook(isolated_bot)

    assert raised.value is error
    assert isolated_bot.load_extension.await_args_list == [
        call(name) for name in _EXTENSIONS[: failed_index + 1]
    ]
    scheduler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_guilds", [False, True], ids=["no-guilds", "multiple-guilds"])
async def test_on_ready_initializes_services_and_guild_caches(
    isolated_bot: MagicMock,
    ready_dependencies: MagicMock,
    capsys: pytest.CaptureFixture[str],
    with_guilds: bool,
) -> None:
    """Подключение загружает данные и создаёт отдельный кэш каждой гильдии."""
    if with_guilds:
        isolated_bot.guilds = [
            _guild(100, "First Guild", [("smile", 123), ("wave", 456)]),
            _guild(200, "Empty Guild", []),
        ]
        isolated_bot.guild_emoji_ids[100] = {"obsolete": "999"}

    await DisBot.on_ready(isolated_bot)

    assert ready_dependencies.mock_calls == [
        call.init_models(),
        call.load_admins(),
        call.load_user_descriptions(),
        call.load_emoji_descriptions(),
        call.create_report_generator(isolated_bot),
    ]
    for step in _READY_STEPS[:-1]:
        getattr(ready_dependencies, step).assert_awaited_once_with()
    assert isolated_bot.report_generator is ready_dependencies.create_report_generator.return_value
    assert isolated_bot.guild_emoji_ids == (
        {100: {"smile": "123", "wave": "456"}, 200: {}} if with_guilds else {}
    )
    assert isolated_bot._build_emoji_cache.call_args_list == [
        call(guild) for guild in isolated_bot.guilds
    ]
    expected_output = "Бот успешно подключился к Discord\n"
    if with_guilds:
        expected_output = (
            "Загружено 2 эмодзи для гильдии First Guild\n"
            "Загружено 0 эмодзи для гильдии Empty Guild\n"
        ) + expected_output
    assert capsys.readouterr().out == expected_output


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_step", _READY_STEPS)
async def test_on_ready_stops_and_preserves_state_after_initialization_failure(
    isolated_bot: MagicMock,
    ready_dependencies: MagicMock,
    capsys: pytest.CaptureFixture[str],
    failed_step: str,
) -> None:
    """При ошибке инициализации последующие шаги и обновление кэша не выполняются."""
    error = RuntimeError("initialization failed")
    getattr(ready_dependencies, failed_step).side_effect = error
    previous_report_generator = isolated_bot.report_generator
    previous_cache = {"old": "789"}
    isolated_bot.guilds = [_guild(100, "Test Guild", [("new", 123)])]
    isolated_bot.guild_emoji_ids[100] = previous_cache

    with pytest.raises(RuntimeError, match="initialization failed") as raised:
        await DisBot.on_ready(isolated_bot)

    assert raised.value is error
    expected_calls = [
        call.init_models(),
        call.load_admins(),
        call.load_user_descriptions(),
        call.load_emoji_descriptions(),
        call.create_report_generator(isolated_bot),
    ]
    failed_index = _READY_STEPS.index(failed_step)
    assert ready_dependencies.mock_calls == expected_calls[: failed_index + 1]
    for step in _READY_STEPS[: min(failed_index + 1, len(_READY_STEPS) - 1)]:
        getattr(ready_dependencies, step).assert_awaited_once_with()
    for step in _READY_STEPS[failed_index + 1 :]:
        getattr(ready_dependencies, step).assert_not_called()
    assert isolated_bot.report_generator is previous_report_generator
    assert isolated_bot.guild_emoji_ids == {100: {"old": "789"}}
    assert isolated_bot.guild_emoji_ids[100] is previous_cache
    isolated_bot._build_emoji_cache.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_on_ready_rebuilds_current_guild_cache_on_reconnection(
    isolated_bot: MagicMock, ready_dependencies: MagicMock
) -> None:
    """Повторное подключение обновляет эмодзи и заменяет генератор отчётов."""
    first_generator = MagicMock()
    second_generator = MagicMock()
    ready_dependencies.create_report_generator.side_effect = [first_generator, second_generator]
    guild = _guild(100, "Test Guild", [("old", 123)])
    isolated_bot.guilds = [guild]

    await DisBot.on_ready(isolated_bot)

    assert isolated_bot.report_generator is first_generator
    assert isolated_bot.guild_emoji_ids == {100: {"old": "123"}}
    guild.emojis = _guild(100, "Test Guild", [("new", 456)]).emojis

    await DisBot.on_ready(isolated_bot)

    assert isolated_bot.report_generator is second_generator
    assert isolated_bot.guild_emoji_ids == {100: {"new": "456"}}
    assert isolated_bot._build_emoji_cache.call_args_list == [call(guild), call(guild)]
    assert ready_dependencies.create_report_generator.call_args_list == [
        call(isolated_bot),
        call(isolated_bot),
    ]
    for step in _READY_STEPS[:-1]:
        assert getattr(ready_dependencies, step).await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("emojis", [[], [("fresh", 123), ("wave", 456)]], ids=["empty", "updated"])
async def test_guild_emojis_update_replaces_only_selected_guild_cache(
    isolated_bot: MagicMock,
    capsys: pytest.CaptureFixture[str],
    emojis: list[tuple[str, int]],
) -> None:
    """Обновление удаляет старые эмодзи, не затрагивая кэш другого сервера."""
    guild = _guild(100, "Test Guild", emojis)
    before = _guild(100, "Test Guild", [("obsolete", 789)]).emojis
    previous_cache = {"obsolete": "789"}
    other_guild_cache = {"other": "999"}
    isolated_bot.guild_emoji_ids = {100: previous_cache, 200: other_guild_cache}

    await DisBot.on_guild_emojis_update(isolated_bot, guild, before, guild.emojis)

    isolated_bot._build_emoji_cache.assert_called_once_with(guild)
    assert isolated_bot.guild_emoji_ids == {
        100: {name: str(emoji_id) for name, emoji_id in emojis},
        200: {"other": "999"},
    }
    assert isolated_bot.guild_emoji_ids[100] is not previous_cache
    assert previous_cache == {"obsolete": "789"}
    assert isolated_bot.guild_emoji_ids[200] is other_guild_cache
    assert capsys.readouterr().out == (
        f"Обновлён кэш эмодзи для гильдии Test Guild: {len(emojis)} эмодзи\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_name", "expected_output"),
    [
        ("on_disconnect", "Бот отключился от Discord\n"),
        ("on_resumed", "Соединение с Discord восстановлено\n"),
    ],
)
async def test_connection_events_print_status(
    isolated_bot: MagicMock,
    capsys: pytest.CaptureFixture[str],
    event_name: str,
    expected_output: str,
) -> None:
    """События отключения и восстановления сообщают о состоянии соединения."""
    await getattr(DisBot, event_name)(isolated_bot)

    assert capsys.readouterr().out == expected_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["обычное сообщение", "!help", "!" + "a" * bot_module.MAX_MESSAGE_LENGTH],
    ids=["plain-text", "command", "overlong-command"],
)
async def test_messages_from_bots_are_ignored_before_other_checks(
    isolated_bot: MagicMock, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """Сообщения ботов не вызывают команды, отчёты и предупреждения о длине."""
    url_check = MagicMock()
    monkeypatch.setattr(bot_module, "contains_only_urls", url_check)
    message = _message(content, author_is_bot=True)

    await DisBot.on_message(isolated_bot, message)

    isolated_bot.process_commands.assert_not_awaited()
    isolated_bot.report_generator.add_message.assert_not_awaited()
    message.channel.send.assert_not_awaited()
    url_check.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   ", "\t\n "], ids=["empty", "spaces", "whitespace"])
async def test_empty_messages_are_ignored_before_url_check(
    isolated_bot: MagicMock, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """Пустой текст и пробельные сообщения не передаются в отчёты и команды."""
    url_check = MagicMock()
    monkeypatch.setattr(bot_module, "contains_only_urls", url_check)
    message = _message(content)

    await DisBot.on_message(isolated_bot, message)

    isolated_bot.process_commands.assert_not_awaited()
    isolated_bot.report_generator.add_message.assert_not_awaited()
    message.channel.send.assert_not_awaited()
    url_check.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "https://example.com/video",
        "www.example.com",
        " https://example.com/one\n http://example.org/two \t",
    ],
    ids=["https", "www", "multiple-urls"],
)
async def test_url_only_messages_are_ignored(isolated_bot: MagicMock, content: str) -> None:
    """Одна или несколько ссылок без другого текста не включаются в отчёты."""
    message = _message(content)

    await DisBot.on_message(isolated_bot, message)

    isolated_bot.report_generator.add_message.assert_not_awaited()
    isolated_bot.process_commands.assert_not_awaited()
    message.channel.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_command", [False, True], ids=["plain-text", "command"])
async def test_message_routing_without_report_generator(
    isolated_bot: MagicMock, is_command: bool
) -> None:
    """Без генератора обычное сообщение пропускается, но команды работают."""
    previous_generator = isolated_bot.report_generator
    isolated_bot.report_generator = None
    message = _message("!help" if is_command else "обычное сообщение")

    await DisBot.on_message(isolated_bot, message)

    if is_command:
        isolated_bot.process_commands.assert_awaited_once_with(message)
    else:
        isolated_bot.process_commands.assert_not_awaited()
    previous_generator.add_message.assert_not_awaited()
    message.channel.send.assert_not_awaited()
    assert isolated_bot.report_generator is None


@pytest.mark.asyncio
async def test_text_containing_url_is_added_to_report(isolated_bot: MagicMock) -> None:
    """Ссылка внутри обычного текста не исключает сообщение из отчётов."""
    message = _message("Посмотрите видео https://example.com/video")

    await DisBot.on_message(isolated_bot, message)

    isolated_bot.report_generator.add_message.assert_awaited_once_with(
        123, message.content, "Test User", 456
    )
    isolated_bot.process_commands.assert_not_awaited()
    message.channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_command_at_maximum_length_is_processed(isolated_bot: MagicMock) -> None:
    """Команда ровно предельной длины выполняется без предупреждения и отчёта."""
    message = _message("!" + "a" * (bot_module.MAX_MESSAGE_LENGTH - 1))
    assert len(message.content) == bot_module.MAX_MESSAGE_LENGTH

    await DisBot.on_message(isolated_bot, message)

    isolated_bot.process_commands.assert_awaited_once_with(message)
    isolated_bot.report_generator.add_message.assert_not_awaited()
    message.channel.send.assert_not_awaited()
