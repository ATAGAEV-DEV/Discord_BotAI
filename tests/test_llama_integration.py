"""Изолированные тесты для app/services/llama_integration.py."""

from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.services import llama_integration
from app.services.llama_integration import LlamaIndexManager


@pytest.fixture
def manager() -> LlamaIndexManager:
    """Создаёт менеджер без инициализации внешних сервисов."""
    instance = object.__new__(LlamaIndexManager)
    instance.db = MagicMock()
    return instance


def _patch_index_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    index: MagicMock,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Подменяет компоненты, создаваемые при построении индекса."""
    document_factory = MagicMock(
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    vector_store = MagicMock()
    storage_context = MagicMock()
    storage_context_factory = MagicMock()
    storage_context_factory.from_defaults.return_value = storage_context
    index_type = MagicMock()
    index_type.from_documents.return_value = index

    monkeypatch.setattr(llama_integration, "Document", document_factory)
    monkeypatch.setattr(llama_integration, "ChromaVectorStore", MagicMock(return_value=vector_store))
    monkeypatch.setattr(llama_integration, "StorageContext", storage_context_factory)
    monkeypatch.setattr(llama_integration, "VectorStoreIndex", index_type)
    return document_factory, vector_store, storage_context, index_type


def _patch_query_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vector_store: MagicMock,
    index: MagicMock,
) -> MagicMock:
    """Подменяет компоненты, создаваемые при поиске контекста."""
    index_type = MagicMock()
    index_type.from_vector_store.return_value = index
    monkeypatch.setattr(llama_integration, "ChromaVectorStore", MagicMock(return_value=vector_store))
    monkeypatch.setattr(llama_integration, "VectorStoreIndex", index_type)
    return index_type


async def _execute_to_thread(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Выполняет переданную функцию вместо запуска рабочего потока."""
    return function(*args, **kwargs)


def test_manager_initialization_configures_llama_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Конструктор создаёт и сохраняет все зависимости LlamaIndex."""
    client = MagicMock()
    embedding_config = {
        "api_key": "embedding-key",
        "base_url": "https://embedding.example/v1",
        "model": "embedding-model",
    }
    embed_model = MagicMock()
    node_parser = MagicMock()
    database = MagicMock()
    embedding_factory = MagicMock(return_value=embed_model)
    parser_factory = MagicMock()
    parser_factory.from_defaults.return_value = node_parser
    settings = MagicMock()

    monkeypatch.setattr(llama_integration, "get_client", MagicMock(return_value=client))
    monkeypatch.setattr(
        llama_integration,
        "get_embedding_config",
        MagicMock(return_value=embedding_config),
    )
    monkeypatch.setattr(llama_integration, "OpenAIEmbedding", embedding_factory)
    monkeypatch.setattr(llama_integration, "SimpleNodeParser", parser_factory)
    monkeypatch.setattr(llama_integration, "chromadb", MagicMock(PersistentClient=MagicMock()))
    llama_integration.chromadb.PersistentClient.return_value = database
    monkeypatch.setattr(llama_integration, "Settings", settings)

    manager = LlamaIndexManager()

    assert manager.custom_client is client
    assert manager.embed_model is embed_model
    assert manager.node_parser is node_parser
    assert manager.db is database
    embedding_factory.assert_called_once_with(
        api_key="embedding-key",
        api_base="https://embedding.example/v1",
        model="embedding-model",
    )
    parser_factory.from_defaults.assert_called_once_with(chunk_size=128, chunk_overlap=16)
    llama_integration.chromadb.PersistentClient.assert_called_once_with(path="./chroma_db")
    assert settings.embed_model is embed_model
    assert settings.node_parser is node_parser


def test_get_server_collection_returns_existing_collection(manager: LlamaIndexManager) -> None:
    """Существующая коллекция возвращается без создания новой."""
    collection = MagicMock()
    manager.db.get_collection.return_value = collection

    result = manager.get_server_collection(123)

    assert result is collection
    manager.db.get_collection.assert_called_once_with("server_123_messages")
    manager.db.create_collection.assert_not_called()


def test_get_server_collection_creates_collection_when_missing(
    manager: LlamaIndexManager,
) -> None:
    """При ошибке получения создаётся коллекция с ожидаемым именем."""
    collection = MagicMock()
    manager.db.get_collection.side_effect = RuntimeError("missing")
    manager.db.create_collection.return_value = collection

    result = manager.get_server_collection(456)

    assert result is collection
    manager.db.get_collection.assert_called_once_with("server_456_messages")
    manager.db.create_collection.assert_called_once_with("server_456_messages")


@pytest.mark.asyncio
async def test_index_messages_builds_dialogue_documents(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сообщения преобразуются в документы с парами user и assistant."""
    index = MagicMock()
    document_factory, vector_store, storage_context, index_type = _patch_index_components(
        monkeypatch,
        index=index,
    )
    collection = MagicMock()
    manager.get_server_collection = MagicMock(return_value=collection)
    to_thread = AsyncMock(side_effect=_execute_to_thread)
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)
    messages = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Without answer"},
        {"role": "system", "content": "ignored again"},
        {"role": "assistant", "content": "Standalone answer"},
    ]

    result = await manager.index_messages(123, messages)

    assert result is index
    assert [call_args.kwargs for call_args in document_factory.call_args_list] == [
        {
            "text": "user: Question\nassistant: Answer",
            "metadata": {"document_type": "message", "server_id": 123},
        },
        {
            "text": "user: Without answer",
            "metadata": {"document_type": "message", "server_id": 123},
        },
        {
            "text": "assistant: Standalone answer",
            "metadata": {"document_type": "message", "server_id": 123},
        },
    ]
    manager.get_server_collection.assert_called_once_with(123)
    llama_integration.ChromaVectorStore.assert_called_once_with(
        chroma_collection=collection,
    )
    llama_integration.StorageContext.from_defaults.assert_called_once_with(
        vector_store=vector_store,
    )
    index_type.from_documents.assert_called_once_with(
        index_type.from_documents.call_args.args[0],
        storage_context=storage_context,
        show_progress=False,
    )
    to_thread.assert_awaited_once_with(
        index_type.from_documents,
        index_type.from_documents.call_args.args[0],
        storage_context=storage_context,
        show_progress=False,
    )


@pytest.mark.asyncio
async def test_index_messages_returns_none_when_no_pairs(
    manager: LlamaIndexManager,
) -> None:
    """При отсутствии user/assistant сообщений индекс не создаётся."""
    result = await manager.index_messages(123, [{"role": "system", "content": "ignored"}])

    assert result is None
    manager.db.get_collection.assert_not_called()
    manager.db.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_index_messages_returns_none_on_timeout(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Таймаут построения индекса обрабатывается без исключения наружу."""
    _patch_index_components(monkeypatch, index=MagicMock())
    manager.get_server_collection = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", AsyncMock(side_effect=TimeoutError))

    result = await manager.index_messages(123, [{"role": "user", "content": "Question"}])

    assert result is None
    assert "Таймаут 25.0 сек." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_index_messages_returns_none_on_unexpected_error(
    manager: LlamaIndexManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Непредвиденная ошибка индексации обрабатывается без исключения наружу."""
    manager.get_server_collection = MagicMock(side_effect=RuntimeError("index failed"))

    result = await manager.index_messages(123, [{"role": "user", "content": "Question"}])

    assert result is None
    assert "Ошибка индексации сообщений: index failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_query_relevant_context_returns_node_texts(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Поиск возвращает текст найденных узлов и использует заданный limit."""
    vector_store = MagicMock()
    index = MagicMock()
    retriever = MagicMock()
    nodes = [SimpleNamespace(text="Первый контекст"), SimpleNamespace(text="Второй контекст")]
    index.as_retriever.return_value = retriever
    retriever.retrieve.return_value = nodes
    to_thread = AsyncMock(side_effect=_execute_to_thread)
    index_type = _patch_query_components(
        monkeypatch,
        vector_store=vector_store,
        index=index,
    )
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)
    collection = MagicMock()
    manager.get_server_collection = MagicMock(return_value=collection)

    result = await manager.query_relevant_context(123, "вопрос", limit=5)

    assert result == ["Первый контекст", "Второй контекст"]
    manager.get_server_collection.assert_called_once_with(123)
    index_type.from_vector_store.assert_called_once_with(vector_store)
    index.as_retriever.assert_called_once_with(similarity_top_k=5)
    retriever.retrieve.assert_called_once_with("вопрос")
    assert to_thread.await_args_list == [
        call(index_type.from_vector_store, vector_store),
        call(retriever.retrieve, "вопрос"),
    ]


@pytest.mark.asyncio
async def test_query_relevant_context_returns_empty_list_on_timeout(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Таймаут поиска контекста возвращает пустой список."""
    manager.get_server_collection = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(llama_integration, "ChromaVectorStore", MagicMock())
    monkeypatch.setattr(
        llama_integration.asyncio,
        "to_thread",
        AsyncMock(side_effect=TimeoutError),
    )

    result = await manager.query_relevant_context(123, "вопрос")

    assert result == []
    assert "Таймаут 30.0 сек." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_query_relevant_context_returns_empty_list_on_unexpected_error(
    manager: LlamaIndexManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Непредвиденная ошибка поиска возвращает пустой список."""
    manager.get_server_collection = MagicMock(side_effect=RuntimeError("query failed"))

    result = await manager.query_relevant_context(123, "вопрос")

    assert result == []
    assert "Ошибка поиска контекста: query failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_index_server_users_deletes_old_data_and_creates_index(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Список пользователей заменяет старый документ и индексируется."""
    index = MagicMock()
    document_factory, vector_store, storage_context, index_type = _patch_index_components(
        monkeypatch,
        index=index,
    )
    collection = MagicMock()
    manager.get_server_collection = MagicMock(return_value=collection)
    to_thread = AsyncMock(side_effect=_execute_to_thread)
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)

    result = await manager.index_server_users(123, ["Alice", "Bob"])

    assert result is index
    assert document_factory.call_args.kwargs == {
        "text": "Список пользователей сервера: Alice, Bob",
        "metadata": {"document_type": "server_users", "server_id": 123},
    }
    collection.delete.assert_called_once_with(where={"document_type": "server_users"})
    llama_integration.StorageContext.from_defaults.assert_called_once_with(
        vector_store=vector_store,
    )
    index_type.from_documents.assert_called_once_with(
        index_type.from_documents.call_args.args[0],
        storage_context=storage_context,
        show_progress=False,
    )
    assert "Обновлен список пользователей сервера 123: 2 пользователей" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_index_server_users_returns_none_on_timeout(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Таймаут удаления или индексации пользователей возвращает None."""
    manager.get_server_collection = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        llama_integration.asyncio,
        "to_thread",
        AsyncMock(side_effect=TimeoutError),
    )

    result = await manager.index_server_users(123, ["Alice"])

    assert result is None
    assert "Таймаут при индексации пользователей сервера 123" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_index_server_users_returns_none_on_unexpected_error(
    manager: LlamaIndexManager,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Непредвиденная ошибка индексации пользователей возвращает None."""
    manager.get_server_collection = MagicMock(side_effect=RuntimeError("users failed"))

    result = await manager.index_server_users(123, ["Alice"])

    assert result is None
    assert "Ошибка индексации пользователей сервера: users failed" in capsys.readouterr().out