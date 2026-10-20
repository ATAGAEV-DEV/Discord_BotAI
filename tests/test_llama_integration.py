"""Изолированные тесты для app/services/llama_integration.py."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.services import llama_integration
from app.services.llama_integration import ChannelHistory, LlamaIndexManager


@pytest.fixture
def manager() -> LlamaIndexManager:
    """Создаёт менеджер без инициализации внешних сервисов."""
    instance = object.__new__(LlamaIndexManager)
    instance.db = MagicMock()
    instance._channel_histories = {}
    instance._channel_server_map = {}
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
    """Конструктор не создаёт global AI client до guild-aware вызова."""
    node_parser = MagicMock()
    database = MagicMock()
    parser_factory = MagicMock()
    parser_factory.from_defaults.return_value = node_parser

    monkeypatch.setattr(llama_integration, "SimpleNodeParser", parser_factory)
    monkeypatch.setattr(llama_integration, "chromadb", MagicMock(PersistentClient=MagicMock()))
    llama_integration.chromadb.PersistentClient.return_value = database

    manager = LlamaIndexManager()

    assert manager.custom_client is None
    assert manager.node_parser is node_parser
    assert manager.db is database
    parser_factory.from_defaults.assert_called_once_with(chunk_size=128, chunk_overlap=16)
    llama_integration.chromadb.PersistentClient.assert_called_once_with(path="./chroma_db")


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
    assert "Ошибка индексации сообщений: RuntimeError" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_query_relevant_context_returns_node_texts(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Поиск возвращает текст найденных узлов и использует заданный limit."""
    vector_store = MagicMock()
    index = MagicMock()
    retriever = MagicMock()
    nodes = [SimpleNamespace(text="Первый контекст", score=1.0), SimpleNamespace(text="Второй контекст", score=1.0)]
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
    assert "Ошибка поиска контекста: RuntimeError" in capsys.readouterr().out


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
    assert "Ошибка индексации пользователей сервера: RuntimeError" in capsys.readouterr().out



# ---------------------------------------------------------------------------
# ChannelHistory tests
# ---------------------------------------------------------------------------


def test_channel_history_starts_empty() -> None:
    """Новая история не содержит сообщений."""
    h = ChannelHistory()
    assert h.get_messages() == []
    assert h.get_texts() == []


def test_channel_history_add_pair_stores_both_messages() -> None:
    """add_pair сохраняет user и assistant сообщения в правильном порядке."""
    h = ChannelHistory()
    h.add_pair({"role": "user", "content": "привет"}, {"role": "assistant", "content": "здравствуй"})
    msgs = h.get_messages()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "привет"}
    assert msgs[1] == {"role": "assistant", "content": "здравствуй"}


def test_channel_history_get_texts_returns_contents() -> None:
    """get_texts возвращает только строки content без role."""
    h = ChannelHistory()
    h.add_pair({"role": "user", "content": "вопрос"}, {"role": "assistant", "content": "ответ"})
    assert h.get_texts() == ["вопрос", "ответ"]


def test_channel_history_respects_max_pairs() -> None:
    """Буфер ограничен max_pairs парами (deque maxlen = max_pairs * 2)."""
    h = ChannelHistory(max_pairs=2)
    for i in range(4):
        h.add_pair({"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"})
    msgs = h.get_messages()
    # Хранятся только последние 2 пары
    assert len(msgs) == 4
    assert msgs[0]["content"] == "q2"
    assert msgs[1]["content"] == "a2"
    assert msgs[2]["content"] == "q3"
    assert msgs[3]["content"] == "a3"


def test_channel_history_clear_empties_messages() -> None:
    """clear() удаляет все сообщения."""
    h = ChannelHistory()
    h.add_pair({"role": "user", "content": "x"}, {"role": "assistant", "content": "y"})
    h.clear()
    assert h.get_messages() == []
    assert h.get_texts() == []


# ---------------------------------------------------------------------------
# LlamaIndexManager channel history management tests
# ---------------------------------------------------------------------------


def test_get_channel_history_creates_new_for_unknown_channel(
    manager: LlamaIndexManager,
) -> None:
    """get_channel_history создаёт новую пустую историю для нового канала."""
    history = manager.get_channel_history(111)
    assert isinstance(history, ChannelHistory)
    assert history.get_messages() == []


def test_get_channel_history_returns_same_object_for_same_channel(
    manager: LlamaIndexManager,
) -> None:
    """get_channel_history возвращает тот же объект при повторном вызове."""
    h1 = manager.get_channel_history(222)
    h2 = manager.get_channel_history(222)
    assert h1 is h2


def test_get_channel_history_stores_server_mapping(
    manager: LlamaIndexManager,
) -> None:
    """server_id сохраняется в _channel_server_map."""
    manager.get_channel_history(333, server_id=99)
    assert manager._channel_server_map[333] == 99



def test_clear_channel_history_clears_messages(
    manager: LlamaIndexManager,
) -> None:
    """clear_channel_history очищает историю указанного канала."""
    history = manager.get_channel_history(555)
    history.add_pair({"role": "user", "content": "x"}, {"role": "assistant", "content": "y"})
    manager.clear_channel_history(555)
    assert history.get_messages() == []


def test_clear_channel_history_noop_for_unknown_channel(
    manager: LlamaIndexManager,
) -> None:
    """clear_channel_history не падает если канал неизвестен."""
    manager.clear_channel_history(9999)  # не должно бросить исключение


def test_clear_server_channel_histories_clears_only_target_server(
    manager: LlamaIndexManager,
) -> None:
    """clear_server_channel_histories очищает только каналы нужного сервера."""
    h1 = manager.get_channel_history(601, server_id=10)
    h2 = manager.get_channel_history(602, server_id=10)
    h3 = manager.get_channel_history(603, server_id=20)

    for h in (h1, h2, h3):
        h.add_pair({"role": "user", "content": "msg"}, {"role": "assistant", "content": "rpl"})

    cleared = manager.clear_server_channel_histories(10)

    assert cleared == 2
    assert h1.get_messages() == []
    assert h2.get_messages() == []
    assert len(h3.get_messages()) == 2


def test_clear_server_channel_histories_returns_zero_for_unknown_server(
    manager: LlamaIndexManager,
) -> None:
    """Возвращает 0 если у сервера нет каналов с историей."""
    manager.get_channel_history(700, server_id=50)
    result = manager.clear_server_channel_histories(99)
    assert result == 0



# ---------------------------------------------------------------------------
# query_relevant_context: score filter + deduplication tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_relevant_context_filters_low_score_nodes(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Узлы с score < RAG_MIN_RELEVANCE_SCORE отфильтровываются."""
    from app.core.config import RAG_MIN_RELEVANCE_SCORE

    low_node = SimpleNamespace(text="low score chunk", score=RAG_MIN_RELEVANCE_SCORE - 0.01)
    high_node = SimpleNamespace(text="high score chunk", score=RAG_MIN_RELEVANCE_SCORE + 0.01)

    vector_store = MagicMock()
    index = MagicMock()
    retriever = MagicMock()
    retriever.retrieve.return_value = [low_node, high_node]
    index.as_retriever.return_value = retriever
    to_thread = AsyncMock(side_effect=_execute_to_thread)

    _patch_query_components(monkeypatch, vector_store=vector_store, index=index)
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)
    manager.get_server_collection = MagicMock(return_value=MagicMock())

    result = await manager.query_relevant_context(123, "запрос")

    assert "high score chunk" in result
    assert "low score chunk" not in result


@pytest.mark.asyncio
async def test_query_relevant_context_passes_node_with_none_score(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Узел с score=None не отфильтровывается (score-фильтр не применяется)."""
    node = SimpleNamespace(text="no score chunk", score=None)

    vector_store = MagicMock()
    index = MagicMock()
    retriever = MagicMock()
    retriever.retrieve.return_value = [node]
    index.as_retriever.return_value = retriever
    to_thread = AsyncMock(side_effect=_execute_to_thread)

    _patch_query_components(monkeypatch, vector_store=vector_store, index=index)
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)
    manager.get_server_collection = MagicMock(return_value=MagicMock())

    result = await manager.query_relevant_context(123, "запрос")

    assert "no score chunk" in result


@pytest.mark.asyncio
async def test_query_relevant_context_deduplicates_against_history_texts(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Чанки совпадающие с history_texts исключаются как дубли."""
    duplicate_node = SimpleNamespace(text="user: привет assistant: здравствуй", score=0.9)
    unique_node = SimpleNamespace(text="совершенно новый контекст", score=0.9)

    vector_store = MagicMock()
    index = MagicMock()
    retriever = MagicMock()
    retriever.retrieve.return_value = [duplicate_node, unique_node]
    index.as_retriever.return_value = retriever
    to_thread = AsyncMock(side_effect=_execute_to_thread)

    _patch_query_components(monkeypatch, vector_store=vector_store, index=index)
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)
    manager.get_server_collection = MagicMock(return_value=MagicMock())

    history_texts = ["user: привет assistant: здравствуй"]

    result = await manager.query_relevant_context(
        123, "запрос", history_texts=history_texts
    )

    assert "совершенно новый контекст" in result
    assert "user: привет assistant: здравствуй" not in result


@pytest.mark.asyncio
async def test_query_relevant_context_no_dedup_without_history(
    manager: LlamaIndexManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без history_texts дедупликация не применяется."""
    node1 = SimpleNamespace(text="первый чанк", score=0.8)
    node2 = SimpleNamespace(text="второй чанк", score=0.8)

    vector_store = MagicMock()
    index = MagicMock()
    retriever = MagicMock()
    retriever.retrieve.return_value = [node1, node2]
    index.as_retriever.return_value = retriever
    to_thread = AsyncMock(side_effect=_execute_to_thread)

    _patch_query_components(monkeypatch, vector_store=vector_store, index=index)
    monkeypatch.setattr(llama_integration.asyncio, "to_thread", to_thread)
    manager.get_server_collection = MagicMock(return_value=MagicMock())

    result = await manager.query_relevant_context(123, "запрос")

    assert result == ["первый чанк", "второй чанк"]
