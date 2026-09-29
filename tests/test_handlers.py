"""Unit-тесты для app/core/handlers.py."""

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
