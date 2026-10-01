"""Изолированные тесты команд описаний пользователей и эмодзи.

Вызовы callback проверяют логику команд, но не проверки прав Discord.
"""

from unittest.mock import AsyncMock, MagicMock, call

import discord
import pytest
from discord.ext import commands

from app.cogs import nicknames
from app.core.bot import DisBot


@pytest.fixture(autouse=True)
def user_cache(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Подменяет ссылку на кэш пользователей в Cog, не меняя настоящий модуль кэша."""
    cache = MagicMock(spec=nicknames.user_descriptions_cache)
    cache.save = AsyncMock()
    cache.remove = AsyncMock()
    cache.load_all = AsyncMock()
    cache.get.return_value = {}
    monkeypatch.setattr(nicknames, "user_descriptions_cache", cache)
    return cache


@pytest.fixture(autouse=True)
def emoji_cache(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Подменяет ссылку на кэш эмодзи в Cog без обращения к реальной БД или кэшу."""
    cache = MagicMock(spec=nicknames.emoji_descriptions_cache)
    cache.save = AsyncMock()
    cache.remove = AsyncMock()
    cache.load_all = AsyncMock()
    cache.get.return_value = {}
    monkeypatch.setattr(nicknames, "emoji_descriptions_cache", cache)
    return cache


@pytest.fixture
def mock_bot() -> MagicMock:
    """Создаёт бот с асинхронным mock регистрации Cog без подключения к Discord."""
    bot = MagicMock(spec=DisBot)
    bot.add_cog = AsyncMock()
    return bot


@pytest.fixture(params=[67890, 98765], ids=["first-guild", "second-guild"])
def mock_ctx(request: pytest.FixtureRequest) -> MagicMock:
    """Проверяет команды на двух серверах с подменённой отправкой сообщений."""
    ctx = MagicMock(spec=commands.Context)
    ctx.guild = MagicMock(spec=discord.Guild)
    ctx.guild.id = request.param
    ctx.send = AsyncMock()
    return ctx


@pytest.fixture
def nicknames_cog(mock_bot: MagicMock) -> nicknames.Nicknames:
    """Создаёт реальный Cog с подменёнными внешними зависимостями."""
    return nicknames.Nicknames(mock_bot)


def test_constructor_stores_bot_without_calling_caches(
    mock_bot: MagicMock, user_cache: MagicMock, emoji_cache: MagicMock
) -> None:
    """Конструктор сохраняет ссылку на бот без загрузки или изменения кэшей."""
    cog = nicknames.Nicknames(mock_bot)

    assert cog.bot is mock_bot
    assert mock_bot.mock_calls == []
    assert user_cache.mock_calls == []
    assert emoji_cache.mock_calls == []


@pytest.mark.asyncio
async def test_setup_registers_nicknames_cog(
    mock_bot: MagicMock, user_cache: MagicMock, emoji_cache: MagicMock
) -> None:
    """Загрузка расширения создаёт Cog и ровно один раз ожидает его регистрацию."""
    await nicknames.setup(mock_bot)

    mock_bot.add_cog.assert_awaited_once()
    cog = mock_bot.add_cog.await_args.args[0]
    assert isinstance(cog, nicknames.Nicknames)
    assert cog.bot is mock_bot
    assert mock_bot.mock_calls == [call.add_cog(cog)]
    assert user_cache.mock_calls == []
    assert emoji_cache.mock_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        "Описание для 'Test User' успешно добавлено!",
        "Описание для 'Test User' успешно обновлено!",
    ],
    ids=["added", "updated"],
)
async def test_desc_add_passes_arguments_and_reports_result(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    result: str,
) -> None:
    """Добавление передаёт ник, полное описание и ID сервера в кэш пользователей."""
    user_cache.save.return_value = result

    await nicknames_cog.desc_add_command.callback(
        nicknames_cog, mock_ctx, nick="Test User", description="Описание с пробелами."
    )

    user_cache.save.assert_awaited_once_with(
        nick="Test User", description="Описание с пробелами.", guild_id=mock_ctx.guild.id
    )
    assert user_cache.mock_calls == [
        call.save(nick="Test User", description="Описание с пробелами.", guild_id=mock_ctx.guild.id)
    ]
    assert emoji_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(f"✅ {result}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    ["Описание для 'Test User' удалено.", "Описание для 'Test User' не найдено."],
    ids=["removed", "not-found"],
)
async def test_desc_remove_passes_arguments_and_reports_result(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    result: str,
) -> None:
    """Удаление передаёт ник и ID сервера, сохраняя сообщение кэша в ответе."""
    user_cache.remove.return_value = result

    await nicknames_cog.desc_remove_command.callback(nicknames_cog, mock_ctx, nick="Test User")

    user_cache.remove.assert_awaited_once_with(nick="Test User", guild_id=mock_ctx.guild.id)
    assert user_cache.mock_calls == [call.remove(nick="Test User", guild_id=mock_ctx.guild.id)]
    assert emoji_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(f"✅ {result}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("descriptions", "expected_message"),
    [
        ({}, "📭 Описания пользователей не найдены."),
        (
            {"Zoya": "Описание Зои"},
            "📋 **Описания пользователей:**\n**Zoya** — Описание Зои",
        ),
        (
            {"Zoya": "Описание Зои", "Alice": "Описание Алисы"},
            "📋 **Описания пользователей:**\n"
            "**Zoya** — Описание Зои\n"
            "**Alice** — Описание Алисы",
        ),
    ],
    ids=["empty", "single", "multiple"],
)
async def test_desc_list_formats_current_guild_descriptions(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    descriptions: dict[str, str],
    expected_message: str,
) -> None:
    """Список сохраняет порядок записей; пустой кэш даёт только ответ об отсутствии."""
    user_cache.get.return_value = dict(descriptions)

    await nicknames_cog.desc_list_command.callback(nicknames_cog, mock_ctx)

    assert user_cache.mock_calls == [call.get(mock_ctx.guild.id)]
    assert user_cache.get.return_value == descriptions
    assert emoji_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(expected_message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("descriptions", "expected_message"),
    [
        ({}, "🔄 Кэш описаний перезагружен! Загружено 0 описаний для этого сервера."),
        (
            {"Zoya": "Описание Зои", "Alice": "Описание Алисы"},
            "🔄 Кэш описаний перезагружен! Загружено 2 описаний для этого сервера.",
        ),
    ],
    ids=["empty", "nonempty"],
)
async def test_desc_reload_awaits_loading_before_counting_current_guild(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    descriptions: dict[str, str],
    expected_message: str,
) -> None:
    """Перезагрузка завершается до чтения, а ответ не учитывает записи другого сервера."""
    loaded = False
    guild_descriptions = {
        mock_ctx.guild.id: {"Old": "Устаревшее описание"},
        mock_ctx.guild.id + 1: {"Other": "Описание с другого сервера"},
    }

    async def load_all() -> None:
        """Заменяет устаревшие данные только при ожидании загрузки."""
        nonlocal loaded
        guild_descriptions[mock_ctx.guild.id] = dict(descriptions)
        loaded = True

    def get(guild_id: int) -> dict[str, str]:
        """Не позволяет прочитать данные до завершения перезагрузки."""
        assert loaded, "load_all() должен завершиться до get()"
        return guild_descriptions[guild_id]

    user_cache.load_all.side_effect = load_all
    user_cache.get.side_effect = get

    await nicknames_cog.desc_reload_command.callback(nicknames_cog, mock_ctx)

    user_cache.load_all.assert_awaited_once_with()
    assert user_cache.mock_calls == [call.load_all(), call.get(mock_ctx.guild.id)]
    assert emoji_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(expected_message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        "Описание эмодзи 'happy_cat' успешно добавлено!",
        "Описание эмодзи 'happy_cat' успешно обновлено!",
    ],
    ids=["added", "updated"],
)
async def test_emoji_add_passes_arguments_and_reports_result(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    result: str,
) -> None:
    """Добавление передаёт имя, полное описание и ID сервера только в кэш эмодзи."""
    emoji_cache.save.return_value = result

    await nicknames_cog.emoji_add_command.callback(
        nicknames_cog, mock_ctx, name="happy_cat", description="Радостный кот с улыбкой."
    )

    emoji_cache.save.assert_awaited_once_with(
        name="happy_cat", description="Радостный кот с улыбкой.", guild_id=mock_ctx.guild.id
    )
    assert emoji_cache.mock_calls == [
        call.save(
            name="happy_cat", description="Радостный кот с улыбкой.", guild_id=mock_ctx.guild.id
        )
    ]
    assert user_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(f"✅ {result}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    ["Описание эмодзи 'happy_cat' удалено.", "Описание эмодзи 'happy_cat' не найдено."],
    ids=["removed", "not-found"],
)
async def test_emoji_remove_passes_arguments_and_reports_result(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    result: str,
) -> None:
    """Удаление передаёт имя и ID сервера, возвращая исходное сообщение кэша эмодзи."""
    emoji_cache.remove.return_value = result

    await nicknames_cog.emoji_remove_command.callback(nicknames_cog, mock_ctx, name="happy_cat")

    emoji_cache.remove.assert_awaited_once_with(name="happy_cat", guild_id=mock_ctx.guild.id)
    assert emoji_cache.mock_calls == [call.remove(name="happy_cat", guild_id=mock_ctx.guild.id)]
    assert user_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(f"✅ {result}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("descriptions", "expected_message"),
    [
        ({}, "📭 Описания эмодзи не найдены."),
        (
            {"happy_cat": "Радостный кот"},
            "📋 **Описания эмодзи:**\n**[e:happy_cat]** — Радостный кот",
        ),
        (
            {"happy_cat": "Радостный кот", "angry_cat": "Сердитый кот"},
            "📋 **Описания эмодзи:**\n"
            "**[e:happy_cat]** — Радостный кот\n"
            "**[e:angry_cat]** — Сердитый кот",
        ),
    ],
    ids=["empty", "single", "multiple"],
)
async def test_emoji_list_formats_current_guild_descriptions(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    descriptions: dict[str, str],
    expected_message: str,
) -> None:
    """Список сохраняет порядок и формат [e:name], а пустой кэш не выводит заголовок."""
    emoji_cache.get.return_value = dict(descriptions)

    await nicknames_cog.emoji_list_command.callback(nicknames_cog, mock_ctx)

    assert emoji_cache.mock_calls == [call.get(mock_ctx.guild.id)]
    assert emoji_cache.get.return_value == descriptions
    assert user_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(expected_message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("descriptions", "expected_message"),
    [
        ({}, "🔄 Кэш эмодзи перезагружен! Загружено 0 описаний."),
        (
            {"happy_cat": "Радостный кот", "angry_cat": "Сердитый кот"},
            "🔄 Кэш эмодзи перезагружен! Загружено 2 описаний.",
        ),
    ],
    ids=["empty", "nonempty"],
)
async def test_emoji_reload_awaits_loading_before_counting_current_guild(
    nicknames_cog: nicknames.Nicknames,
    mock_ctx: MagicMock,
    user_cache: MagicMock,
    emoji_cache: MagicMock,
    descriptions: dict[str, str],
    expected_message: str,
) -> None:
    """Перезагрузка ожидается до чтения, а счётчик описаний ограничен текущим сервером."""
    loaded = False
    guild_descriptions = {
        mock_ctx.guild.id: {"old": "Устаревшее описание"},
        mock_ctx.guild.id + 1: {"other": "Описание с другого сервера"},
    }

    async def load_all() -> None:
        """Заменяет устаревшие данные только при ожидании загрузки."""
        nonlocal loaded
        guild_descriptions[mock_ctx.guild.id] = dict(descriptions)
        loaded = True

    def get(guild_id: int) -> dict[str, str]:
        """Не позволяет прочитать данные до завершения перезагрузки."""
        assert loaded, "load_all() должен завершиться до get()"
        return guild_descriptions[guild_id]

    emoji_cache.load_all.side_effect = load_all
    emoji_cache.get.side_effect = get

    await nicknames_cog.emoji_reload_command.callback(nicknames_cog, mock_ctx)

    emoji_cache.load_all.assert_awaited_once_with()
    assert emoji_cache.mock_calls == [call.load_all(), call.get(mock_ctx.guild.id)]
    assert user_cache.mock_calls == []
    mock_ctx.send.assert_awaited_once_with(expected_message)
