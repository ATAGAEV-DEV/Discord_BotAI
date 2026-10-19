import asyncio
import time
from collections import deque
from typing import Any

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.ai_config import get_embedding_model
from app.core.config import CONVERSATION_MAX_PAIRS, CONVERSATION_TTL, RAG_MIN_RELEVANCE_SCORE
from app.data import guild_ai_config_cache

QUERY_TIMEOUT = 30.0
INDEX_TIMEOUT = 25.0


class ChannelHistory:
    """Краткосрочная память одного Discord-канала.

    Хранит последние ``max_pairs`` пар (user, assistant) в виде плоского
    deque словарей ``{"role": ..., "content": ...}``. Полностью независима
    от LlamaIndex и ChromaDB — её отсутствие или сброс не влияет на RAG.
    """

    def __init__(self, max_pairs: int = CONVERSATION_MAX_PAIRS) -> None:
        """Инициализирует буфер истории."""
        self.max_pairs = max_pairs
        # maxlen = max_pairs * 2, т.к. каждая пара = 2 сообщения
        self._messages: deque[dict[str, str]] = deque(maxlen=max_pairs * 2)
        self._last_activity: float = time.monotonic()

    def add_pair(self, user_msg: dict[str, str], assistant_msg: dict[str, str]) -> None:
        """Добавляет пару (user, assistant) в буфер."""
        self._messages.append(user_msg)
        self._messages.append(assistant_msg)
        self._last_activity = time.monotonic()

    def get_messages(self) -> list[dict[str, str]]:
        """Возвращает все сообщения в хронологическом порядке."""
        return list(self._messages)

    def get_texts(self) -> list[str]:
        """Возвращает только текстовое содержимое всех сообщений."""
        return [m["content"] for m in self._messages]

    def clear(self) -> None:
        """Сбрасывает историю."""
        self._messages.clear()
        self._last_activity = time.monotonic()

    def is_expired(self, ttl: float = CONVERSATION_TTL) -> bool:
        """Возвращает True, если канал неактивен дольше TTL."""
        return (time.monotonic() - self._last_activity) > ttl


class LlamaIndexManager:
    """Управляет интеграцией с LlamaIndex и ChromaDB.

    Строит векторные индексы сообщений Discord-сервера.
    ``ChannelHistory`` — независимый in-memory слой краткосрочной памяти;
    его отсутствие или сброс не влияет на RAG-индекс ChromaDB.
    """

    def __init__(self) -> None:
        """Инициализирует менеджер LlamaIndex."""
        self.custom_client = None
        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=128, chunk_overlap=16)
        self.db = chromadb.PersistentClient(path="./chroma_db")
        # per-channel краткосрочная память: channel_id → ChannelHistory
        self._channel_histories: dict[int, ChannelHistory] = {}
        # маппинг channel_id → server_id для очистки по серверу
        self._channel_server_map: dict[int, int] = {}

    def _get_embedding_model(self, server_id: int | None) -> OpenAIEmbedding | None:
        """Возвращает embedding model конкретной гильдии после bootstrap.

        До загрузки bot cache изолированные unit-тесты могут строить mock-index
        без реального клиента. В работающем боте ``is_loaded()`` истинно после
        ``on_ready``, поэтому cache miss остаётся fail-closed.
        """
        if server_id is not None and not guild_ai_config_cache.is_loaded():
            return None
        return get_embedding_model(server_id)

    def get_server_collection(self, server_id: int) -> Any:
        """Получить или создать коллекцию для сервера."""
        collection_name = f"server_{server_id}_messages"
        try:
            collection = self.db.get_collection(collection_name)
        except Exception:
            collection = self.db.create_collection(collection_name)
        return collection

    def get_channel_history(self, channel_id: int, server_id: int | None = None) -> ChannelHistory:
        """Возвращает ChannelHistory для канала, создаёт при первом обращении.

        Если история истекла по TTL — сбрасывает и возвращает чистую.
        ``server_id`` сохраняется для последующей очистки по серверу.
        """
        history = self._channel_histories.get(channel_id)
        if history is None:
            history = ChannelHistory()
            self._channel_histories[channel_id] = history
        elif history.is_expired():
            history.clear()

        if server_id is not None:
            self._channel_server_map[channel_id] = server_id

        return history

    def clear_channel_history(self, channel_id: int) -> None:
        """Явно сбрасывает историю одного канала."""
        history = self._channel_histories.get(channel_id)
        if history is not None:
            history.clear()

    def clear_server_channel_histories(self, server_id: int) -> int:
        """Сбрасывает историю всех каналов указанного сервера.

        Возвращает количество очищенных каналов.
        """
        cleared = 0
        for channel_id, sid in self._channel_server_map.items():
            if sid == server_id:
                history = self._channel_histories.get(channel_id)
                if history is not None:
                    history.clear()
                    cleared += 1
        return cleared

    async def index_messages(self, server_id: int, messages: list[dict[str, Any]]) -> Any:
        """Индексировать сообщения сервера как диалоговые пары user+assistant."""
        try:
            pairs = []
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg.get("role") == "user":
                    pair_text = f"user: {msg['content']}"
                    if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                        pair_text += f"\nassistant: {messages[i + 1]['content']}"
                        i += 2
                    else:
                        i += 1
                    pairs.append(pair_text)
                elif msg.get("role") == "assistant":
                    pairs.append(f"assistant: {msg['content']}")
                    i += 1
                else:
                    i += 1

            if not pairs:
                return None

            documents = [
                Document(
                    text=text,
                    metadata={"document_type": "message", "server_id": server_id},
                )
                for text in pairs
            ]

            collection = self.get_server_collection(server_id)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            embed_model = self._get_embedding_model(server_id)
            index_kwargs = {
                "storage_context": storage_context,
                "show_progress": False,
            }
            if embed_model is not None:
                index_kwargs["embed_model"] = embed_model

            index = await asyncio.wait_for(
                asyncio.to_thread(
                    VectorStoreIndex.from_documents,
                    documents,
                    **index_kwargs,
                ),
                timeout=INDEX_TIMEOUT,
            )
            return index
        except TimeoutError:
            print(f"Таймаут {INDEX_TIMEOUT} сек. при индексации сообщений сервера {server_id}")
            return None
        except Exception as exc:
            print(f"Ошибка индексации сообщений: {type(exc).__name__}")
            return None

    async def query_relevant_context(
        self,
        server_id: int,
        query: str,
        limit: int = 8,
        history_texts: list[str] | None = None,
    ) -> list[str]:
        """Найти релевантный контекст для запроса на сервере.

        Фрагменты с score < MIN_RELEVANCE_SCORE отбрасываются.
        Фрагменты, чей нормализованный текст содержится в тексте любого
        сообщения из ``history_texts`` (или наоборот), исключаются как дубли —
        они уже есть в краткосрочной памяти канала.
        """
        try:
            collection = self.get_server_collection(server_id)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            embed_model = self._get_embedding_model(server_id)
            index_kwargs: dict[str, Any] = {}
            if embed_model is not None:
                index_kwargs["embed_model"] = embed_model
            index = await asyncio.wait_for(
                asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store, **index_kwargs),
                timeout=QUERY_TIMEOUT,
            )

            retriever = index.as_retriever(similarity_top_k=limit)
            nodes = await asyncio.wait_for(
                asyncio.to_thread(retriever.retrieve, query), timeout=QUERY_TIMEOUT
            )

            # Нормализованные тексты из краткосрочной истории для дедупликации
            def _norm(s: str) -> str:
                return s.lower().split()

            history_norm: list[str] = []
            if history_texts:
                history_norm = [" ".join(_norm(t)) for t in history_texts]

            relevant_contexts: list[str] = []
            for node in nodes:
                # Фильтр по score релевантности
                score = node.score
                if score is not None and score < RAG_MIN_RELEVANCE_SCORE:
                    continue

                chunk = node.text
                chunk_norm = " ".join(_norm(chunk))

                # Дедупликация: пропускаем если chunk уже есть в истории
                is_duplicate = any(
                    chunk_norm in h or h in chunk_norm
                    for h in history_norm
                )
                if is_duplicate:
                    continue

                relevant_contexts.append(chunk)

            return relevant_contexts
        except TimeoutError:
            print(f"Таймаут {QUERY_TIMEOUT} сек. при поиске контекста для сервера {server_id}")
            return []
        except Exception as exc:
            print(f"Ошибка поиска контекста: {type(exc).__name__}")
            return []

    async def index_server_users(self, server_id: int, users: list[str]) -> Any:
        """Индексировать список пользователей сервера с использованием метаданных."""
        try:
            collection = self.get_server_collection(server_id)
            await asyncio.wait_for(
                asyncio.to_thread(collection.delete, where={"document_type": "server_users"}),
                timeout=5.0,
            )
            users_text = f"Список пользователей сервера: {', '.join(users)}"
            document = Document(
                text=users_text, metadata={"document_type": "server_users", "server_id": server_id}
            )

            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            embed_model = self._get_embedding_model(server_id)
            index_kwargs = {
                "storage_context": storage_context,
                "show_progress": False,
            }
            if embed_model is not None:
                index_kwargs["embed_model"] = embed_model

            index = await asyncio.wait_for(
                asyncio.to_thread(
                    VectorStoreIndex.from_documents,
                    [document],
                    **index_kwargs,
                ),
                timeout=INDEX_TIMEOUT,
            )

            print(f"Обновлен список пользователей сервера {server_id}: {len(users)} пользователей")
            return index
        except TimeoutError:
            print(f"Таймаут при индексации пользователей сервера {server_id}")
            return None
        except Exception as exc:
            print(f"Ошибка индексации пользователей сервера: {type(exc).__name__}")
            return None
