"""Unit-тесты для app/core/handlers.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import AI_GENERATE_TIMEOUT
from app.core.handlers import ai_generate, clear_server_history


class TestClearServerHistory:
    """Тесты для функции clear_server_history."""

    @pytest.mark.asyncio
    @patch("app.core.handlers.llama_manager")
    async def test_deletes_non_user_documents(self, mock_llama: MagicMock) -> None:
        """Удаляет документы, не являющиеся server_users."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc1", "doc2", "doc3"],
            "metadatas": [
                {"document_type": "message"},
                {"document_type": "server_users"},
                {"document_type": "context"},
            ],
        }
        mock_llama.get_server_collection.return_value = mock_collection

        result = await clear_server_history(12345)
        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc3"])
        assert "2" in result
        assert "Удалено" in result

    @pytest.mark.asyncio
    @patch("app.core.handlers.llama_manager")
    async def test_empty_collection(self, mock_llama: MagicMock) -> None:
        """Пустая коллекция — сообщение о пустом индексе."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_llama.get_server_collection.return_value = mock_collection

        result = await clear_server_history(12345)
        assert "пуст" in result

    @pytest.mark.asyncio
    @patch("app.core.handlers.llama_manager")
    async def test_only_server_users(self, mock_llama: MagicMock) -> None:
        """Только документы server_users — ничего не удаляется."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["doc1"],
            "metadatas": [{"document_type": "server_users"}],
        }
        mock_llama.get_server_collection.return_value = mock_collection

        result = await clear_server_history(12345)
        mock_collection.delete.assert_not_called()
        assert "нет документов" in result

    @pytest.mark.asyncio
    @patch("app.core.handlers.llama_manager")
    async def test_exception_handling(self, mock_llama: MagicMock) -> None:
        """При ошибке — исключение пробрасывается наружу."""
        mock_llama.get_server_collection.side_effect = Exception("DB Error")

        with pytest.raises(Exception, match="DB Error"):
            await clear_server_history(12345)


@pytest.mark.asyncio
async def test_ai_generate_uses_shared_api_timeout() -> None:
    """API-вызов использует тот же таймаут, что и вся генерация."""
    with (
        patch("app.core.handlers.llama_manager") as mock_llama,
        patch("app.core.handlers.user_prompt", return_value="system prompt"),
        patch("app.core.handlers.user_descriptions_cache.get", return_value={}),
        patch("app.core.handlers.get_client") as mock_get_client,
    ):
        mock_llama.query_relevant_context = AsyncMock(return_value=[])
        mock_create = AsyncMock(side_effect=TimeoutError)
        mock_get_client.return_value.chat.completions.create = mock_create

        result = await ai_generate("test", 12345, "test_user")

    mock_create.assert_awaited_once()
    assert mock_create.await_args.kwargs["timeout"] == AI_GENERATE_TIMEOUT
    assert result == f"⏳ Запрос не обработан за {AI_GENERATE_TIMEOUT} секунд. Попробуйте позже."


@pytest.mark.asyncio
async def test_ai_generate_processes_context_and_response() -> None:
    """Успешная генерация обогащает контекст, очищает ответ и индексирует диалог."""
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="raw **answer**"))]
    )

    with (
        patch("app.core.handlers.llama_manager") as mock_llama,
        patch("app.core.handlers.user_prompt", return_value="system prompt") as mock_user_prompt,
        patch(
            "app.core.handlers.user_descriptions_cache.get",
            return_value={"Alice": "Алиса"},
        ) as mock_descriptions,
        patch(
            "app.core.handlers.enrich_users_context",
            return_value=["контекст сервера"],
        ) as mock_enrich,
        patch("app.core.handlers.get_client") as mock_get_client,
        patch("app.core.handlers.get_model", return_value="test-model"),
        patch(
            "app.core.handlers.discord.utils.remove_markdown",
            return_value="clean answer",
        ) as mock_remove_markdown,
        patch(
            "app.core.handlers.replace_emojis",
            return_value="clean answer 😀",
        ) as mock_replace_emojis,
        patch("app.core.handlers.strip_emoji", return_value="indexed answer") as mock_strip,
        patch("app.core.handlers.count_tokens", return_value=7) as mock_count_tokens,
        patch("app.core.handlers.asyncio.create_task") as mock_create_task,
    ):
        mock_llama.query_relevant_context = AsyncMock(return_value=["raw context"])
        mock_llama.index_messages = AsyncMock()
        mock_create_task.side_effect = lambda coroutine: coroutine.close()
        mock_create = AsyncMock(return_value=completion)
        mock_get_client.return_value.chat.completions.create = mock_create

        result = await ai_generate(
            "Как дела?",
            12345,
            "Alice",
            limit=7,
            emoji_ids={"smile": "100"},
        )

    assert result == "clean answer 😀"
    mock_user_prompt.assert_called_once_with(
        "Alice", 12345, emoji_ids={"smile": "100"}, text="Как дела?", mentioned_names=[]
    )
    mock_descriptions.assert_called_once_with(12345)
    mock_llama.query_relevant_context.assert_awaited_once_with(12345, "Как дела?", limit=7)
    mock_enrich.assert_called_once_with(["raw context"], {"Alice": "Алиса"})
    mock_create.assert_awaited_once()
    assert mock_create.await_args.kwargs["model"] == "test-model"
    messages = mock_create.await_args.kwargs["messages"]
    assert [message["role"] for message in messages] == ["system", "system", "user"]
    assert messages[1]["content"].endswith("контекст сервера")
    assert messages[2]["content"] == "[Пользователь: Alice] Как дела?"
    mock_remove_markdown.assert_called_once_with("raw **answer**")
    mock_replace_emojis.assert_called_once_with("clean answer", {"smile": "100"})
    mock_strip.assert_called_once_with("clean answer 😀")
    mock_count_tokens.assert_called_once()
    mock_llama.index_messages.assert_called_once_with(
        12345,
        [
            {"role": "user", "content": "[Пользователь: Alice] Как дела?"},
            {"role": "assistant", "content": "indexed answer"},
        ],
    )
    mock_create_task.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("known_user", [False, True], ids=["unknown-user", "known-user"])
@pytest.mark.parametrize(
    ("server_id", "emoji_ids", "expected_list"),
    [
        (123, {"Shared": "111", "NoDescription": "333"}, "[e:Shared] — первый сервер"),
        (456, {"Shared": "222"}, "[e:Shared] — второй сервер"),
        (123, {}, ""),
        (123, None, ""),
        (123, {"Shared": ""}, ""),
        (123, {"NoDescription": "333"}, ""),
        (123, {"shared": "111"}, ""),
        (789, {"Shared": "111"}, ""),
        (None, {}, ""),
        (None, {"Shared": "111"}, ""),
    ],
    ids=[
        "first-guild",
        "second-guild",
        "empty-ids",
        "missing-ids",
        "empty-id-value",
        "no-matching-descriptions",
        "case-mismatch",
        "no-guild-descriptions",
        "private-message",
        "private-message-with-ids",
    ],
)
async def test_ai_generate_sends_only_current_guild_emojis(
    monkeypatch: pytest.MonkeyPatch,
    server_id: int | None,
    emoji_ids: dict[str, str] | None,
    expected_list: str,
    known_user: bool,
) -> None:
    """Настоящий системный промпт в запросе к AI содержит только доступные эмодзи сервера."""
    from app.data import emoji_descriptions_cache, user_descriptions_cache

    monkeypatch.setattr(
        user_descriptions_cache,
        "_cache",
        {123: {"Alice": "описание автора"}, 456: {"Alice": "описание автора"}} if known_user else {},
    )
    monkeypatch.setattr(
        emoji_descriptions_cache,
        "_cache",
        {
            123: {"Shared": "первый сервер", "Deleted": "устаревшее описание"},
            456: {
                "Shared": "второй сервер",
                "Foreign": "чужой эмодзи",
                "NoDescription": "описание только на другом сервере",
            },
        },
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ."))]
    )

    with (
        patch("app.core.handlers.llama_manager") as mock_llama,
        patch("app.core.handlers.get_client") as mock_get_client,
        patch("app.core.handlers.get_model", return_value="test-model"),
        patch("app.core.handlers.asyncio.create_task") as mock_create_task,
    ):
        mock_llama.query_relevant_context = AsyncMock(return_value=[])
        mock_llama.index_messages = AsyncMock()
        mock_create_task.side_effect = lambda coroutine: coroutine.close()
        mock_create = AsyncMock(return_value=completion)
        mock_get_client.return_value.chat.completions.create = mock_create

        result = await ai_generate("Как дела?", server_id, "Alice", emoji_ids=emoji_ids)

    assert result == "Ответ."
    mock_create.assert_awaited_once()
    messages = mock_create.await_args.kwargs["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    system_prompt = messages[0]["content"]
    if expected_list:
        assert system_prompt.count("Использовать только эмодзи сервера из списка ниже.") == 1
        assert system_prompt.count("Правила формата эмодзи:") == 1
        assert system_prompt.count("Доступные эмодзи:") == 1
        emoji_list = (
            system_prompt.split("Доступные эмодзи:", 1)[1]
            .split("Информация по пользователям", 1)[0]
            .strip()
        )
        assert emoji_list == expected_list
    else:
        assert "эмодзи" not in system_prompt
        assert "[e:" not in system_prompt

    if known_user and server_id in (123, 456):
        assert system_prompt.count("Информация по пользователям") == 1
        assert system_prompt.count("- Alice: описание автора") == 1
    else:
        assert "Информация по пользователям" not in system_prompt

    assert "{emoji_section}" not in system_prompt
    assert "{emoji_list}" not in system_prompt
    assert "{user_info}" not in system_prompt
    assert messages[1]["content"] == "[Пользователь: Alice] Как дела?"
    mock_llama.query_relevant_context.assert_awaited_once_with(server_id, "Как дела?", limit=15)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_context", [False, True], ids=["no-history", "with-history"])
@pytest.mark.parametrize(
    ("server_id", "local_description", "expected_author"),
    [
        (123, "автор первого сервера", "автор первого сервера"),
        (456, None, "автор второго сервера"),
        (123, "", ""),
        (123, " \t\n ", ""),
        (123, None, ""),
        (789, None, ""),
        (999, None, ""),
        (None, None, ""),
    ],
    ids=[
        "first-guild",
        "second-guild",
        "empty-description",
        "whitespace-description",
        "foreign-author-only",
        "unknown-guild",
        "empty-guild",
        "dm",
    ],
)
async def test_ai_generate_isolates_descriptions_in_both_system_messages(
    monkeypatch: pytest.MonkeyPatch,
    server_id: int | None,
    local_description: str | None,
    expected_author: str,
    has_context: bool,
) -> None:
    """Автор добавляется независимо от истории, а оба system-блока изолированы сервером."""
    from app.data import emoji_descriptions_cache, user_descriptions_cache

    cache = {
        0: {"Alice": "старое описание автора", "LegacyOnly": "старые общие данные"},
        123: {"Bob": "контекст первого сервера", "Empty": "", "Blank": " \t\n "},
        456: {
            "Alice": "автор второго сервера",
            "Bob": "контекст второго сервера",
            "ForeignOnly": "данные второго сервера",
        },
        999: {},
    }
    if local_description is not None:
        cache[123]["Alice"] = local_description
    original_cache = {guild: descriptions.copy() for guild, descriptions in cache.items()}
    monkeypatch.setattr(user_descriptions_cache, "_cache", cache)
    monkeypatch.setattr(emoji_descriptions_cache, "_cache", {})
    users_context = "Список пользователей сервера: Bob, ForeignOnly, LegacyOnly, Empty, Blank"
    contexts = [users_context, "Обычная история"] if has_context else []
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ."))]
    )

    with (
        patch("app.core.handlers.llama_manager") as mock_llama,
        patch("app.core.handlers.get_client") as mock_get_client,
        patch("app.core.handlers.get_model", return_value="test-model"),
        patch("app.core.handlers.asyncio.create_task") as mock_create_task,
        patch(
            "app.core.handlers.user_descriptions_cache.get_all",
            side_effect=AssertionError("AI не должен читать описания со всех серверов"),
        ),
    ):
        mock_llama.query_relevant_context = AsyncMock(return_value=contexts)
        mock_llama.index_messages = AsyncMock()
        mock_create_task.side_effect = lambda coroutine: coroutine.close()
        mock_create = AsyncMock(return_value=completion)
        mock_get_client.return_value.chat.completions.create = mock_create

        result = await ai_generate("Как дела?", server_id, "Alice")

    assert result == "Ответ."
    mock_create.assert_awaited_once()
    messages = mock_create.await_args.kwargs["messages"]
    assert [message["role"] for message in messages] == (
        ["system", "system", "user"] if has_context else ["system", "user"]
    )
    author_prompt = messages[0]["content"]
    if expected_author:
        assert author_prompt.count(f"- Alice: {expected_author}") == 1
    else:
        assert "Информация по пользователям" not in author_prompt
        assert "- Alice:" not in author_prompt
    assert "эмодзи" not in author_prompt

    if has_context:
        expected_users = {
            123: "Bob: контекст первого сервера; ForeignOnly; LegacyOnly; Empty; Blank",
            456: (
                "Bob: контекст второго сервера; ForeignOnly: данные второго сервера; "
                "LegacyOnly; Empty; Blank"
            ),
        }.get(server_id, "Bob; ForeignOnly; LegacyOnly; Empty; Blank")
        assert messages[1]["content"] == (
            "Релевантный контекст из истории сервера:\n"
            f"Список пользователей сервера: {expected_users}\nОбычная история"
        )
        assert "Alice" not in messages[1]["content"]

    all_system_content = "\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    assert "старое описание автора" not in all_system_content
    assert "старые общие данные" not in all_system_content
    assert "Empty:" not in all_system_content
    assert "Blank:" not in all_system_content
    assert user_descriptions_cache._cache == original_cache
    assert messages[-1]["content"] == "[Пользователь: Alice] Как дела?"


@pytest.mark.asyncio
@pytest.mark.parametrize("mention_form", ["native", "nickname-native", "literal", "mixed"])
@pytest.mark.parametrize("has_context", [False, True], ids=["empty-history", "with-history"])
@pytest.mark.parametrize(
    ("server_id", "expected_user_lines"),
    [
        (123, ["- Alice: автор первого сервера", "- Ded: Дед первого сервера"]),
        (
            456,
            [
                "- Alice: автор второго сервера",
                "- Ded: Дед второго сервера",
                "- ForeignOnly: пользователь второго сервера",
            ],
        ),
        (789, []),
        (999, []),
        (None, []),
    ],
    ids=["first-guild", "second-guild", "missing-guild", "empty-guild", "dm"],
)
async def test_ai_generate_resolves_mentions_and_uses_guild_descriptions_without_history(
    monkeypatch: pytest.MonkeyPatch,
    server_id: int | None,
    expected_user_lines: list[str],
    has_context: bool,
    mention_form: str,
) -> None:
    """Настоящий prompt получает описания упоминаний независимо от RAG и без утечки."""
    from app.data import emoji_descriptions_cache, user_descriptions_cache

    cache = {
        0: {
            "Alice": "общее описание автора",
            "Ded": "общее описание Деда",
            "LegacyOnly": "старые общие данные",
            "NoDescription": "общее описание без серверного",
        },
        123: {
            "Alice": "автор первого сервера",
            "Ded": "Дед первого сервера",
            "Empty": "",
            "Blank": " \t\n ",
            "MetadataOnly": "пользователь вне текста",
            "99999": "неизвестный ID не является username",
        },
        456: {
            "Alice": "автор второго сервера",
            "Ded": "Дед второго сервера",
            "ForeignOnly": "пользователь второго сервера",
        },
        999: {},
    }
    original_cache = {guild_id: descriptions.copy() for guild_id, descriptions in cache.items()}
    monkeypatch.setattr(user_descriptions_cache, "_cache", cache)
    monkeypatch.setattr(emoji_descriptions_cache, "_cache", {})
    token = {
        "native": "<@321>",
        "nickname-native": "<@!321>",
        "literal": "@Ded",
        "mixed": "<@321> @Ded",
    }[mention_form]
    no_description_token = "@NoDescription" if mention_form == "literal" else "<@987>"
    suffix = "@Alice @Empty @Blank @ForeignOnly @LegacyOnly @Unknown <@99999> <@&321> <#321>"
    text = f"Кто такой {token}, {token}? {no_description_token} {suffix}"
    resolved_token = "@Ded @Ded" if mention_form == "mixed" else "@Ded"
    expected_text = f"Кто такой {resolved_token}, {resolved_token}? @NoDescription {suffix}"
    mentions = (
        {} if mention_form == "literal" else {321: "Ded", 987: "NoDescription", 1: "MetadataOnly"}
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ."))]
    )

    with (
        patch("app.core.handlers.llama_manager") as mock_llama,
        patch("app.core.handlers.get_client") as mock_get_client,
        patch("app.core.handlers.get_model", return_value="test-model"),
        patch("app.core.handlers.asyncio.create_task") as mock_create_task,
    ):
        mock_llama.query_relevant_context = AsyncMock(
            return_value=["Обычная история"] if has_context else []
        )
        mock_llama.index_messages = AsyncMock()
        mock_create_task.side_effect = lambda coroutine: coroutine.close()
        mock_create = AsyncMock(return_value=completion)
        mock_get_client.return_value.chat.completions.create = mock_create

        result = await ai_generate(text, server_id, "Alice", limit=7, mentions=mentions)

    assert result == "Ответ."
    mock_create.assert_awaited_once()
    messages = mock_create.await_args.kwargs["messages"]
    assert [message["role"] for message in messages] == (
        ["system", "system", "user"] if has_context else ["system", "user"]
    )
    system_prompt = messages[0]["content"]
    assert [line for line in system_prompt.splitlines() if line.startswith("- ")] == (
        expected_user_lines
    )
    if expected_user_lines:
        assert system_prompt.count("Информация по пользователям") == 1
    else:
        assert "Информация по пользователям" not in system_prompt
    for excluded in ("Empty", "Blank", "LegacyOnly", "Unknown", "NoDescription", "MetadataOnly"):
        assert excluded not in system_prompt
    assert "неизвестный ID" not in system_prompt
    assert "общее описание" not in system_prompt
    if has_context:
        assert messages[1]["content"] == (
            "Релевантный контекст из истории сервера:\nОбычная история"
        )
    expected_user_message = {"role": "user", "content": f"[Пользователь: Alice] {expected_text}"}
    assert messages[-1] == expected_user_message
    mock_llama.query_relevant_context.assert_awaited_once_with(server_id, expected_text, limit=7)
    mock_llama.index_messages.assert_called_once_with(
        server_id,
        [expected_user_message, {"role": "assistant", "content": "Ответ."}],
    )
    mock_create_task.assert_called_once()
    assert user_descriptions_cache._cache == original_cache


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [":e:Gachi1", ":Gachi1:", "e:Gachi1"])
async def test_ai_generate_repairs_cached_emoji_before_indexing(marker: str) -> None:
    """Настоящая обработка исправляет известный маркер и убирает эмодзи из индекса."""
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=f"Ответ **текст** {marker} и :e:unknown")
            )
        ]
    )

    with (
        patch("app.core.handlers.llama_manager") as mock_llama,
        patch("app.core.handlers.user_prompt", return_value="system prompt"),
        patch("app.core.handlers.user_descriptions_cache.get", return_value={}),
        patch("app.core.handlers.get_client") as mock_get_client,
        patch("app.core.handlers.get_model", return_value="test-model"),
        patch("app.core.handlers.count_tokens", return_value=7),
        patch("app.core.handlers.asyncio.create_task") as mock_create_task,
    ):
        mock_llama.query_relevant_context = AsyncMock(return_value=[])
        mock_llama.index_messages = AsyncMock()
        mock_create_task.side_effect = lambda coroutine: coroutine.close()
        mock_create = AsyncMock(return_value=completion)
        mock_get_client.return_value.chat.completions.create = mock_create

        result = await ai_generate(
            "Как дела?", 12345, "Alice", emoji_ids={"Gachi1": "469464559959277578"}
        )

    assert result == "Ответ текст <:Gachi1:469464559959277578> и :e:unknown"
    mock_create.assert_awaited_once()
    mock_llama.index_messages.assert_called_once_with(
        12345,
        [
            {"role": "user", "content": "[Пользователь: Alice] Как дела?"},
            {"role": "assistant", "content": "Ответ текст  и :e:unknown"},
        ],
    )
    mock_create_task.assert_called_once()
