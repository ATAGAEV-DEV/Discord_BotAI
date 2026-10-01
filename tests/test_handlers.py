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
        patch("app.core.handlers.user_descriptions_cache.get_all", return_value={}),
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
        patch("app.core.handlers.user_prompt", return_value="system prompt"),
        patch(
            "app.core.handlers.user_descriptions_cache.get_all",
            return_value={"Alice": "Алиса"},
        ),
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
        patch("app.core.handlers.user_descriptions_cache.get_all", return_value={}),
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
