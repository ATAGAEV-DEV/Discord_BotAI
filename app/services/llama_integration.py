import asyncio
from typing import Any

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.ai_config import get_client, get_provider_config

QUERY_TIMEOUT = 30.0
INDEX_TIMEOUT = 25.0


class LlamaIndexManager:
    """Управляет интеграцией с LlamaIndex и ChromaDB.

    Строит векторные индексы сообщений Discord-сервера.
    """

    def __init__(self) -> None:
        """Инициализирует менеджер LlamaIndex."""
        self.custom_client = get_client()

        config = get_provider_config()
        self.embed_model = OpenAIEmbedding(
            api_key=config["api_key"],
            api_base=config["base_url"],
            model="text-embedding-3-large",
        )

        self.node_parser = SimpleNodeParser.from_defaults(chunk_size=128, chunk_overlap=16)
        self.db = chromadb.PersistentClient(path="./chroma_db")

        Settings.embed_model = self.embed_model
        Settings.node_parser = self.node_parser

    def get_server_collection(self, server_id: int) -> Any:
        """Получить или создать коллекцию для сервера."""
        collection_name = f"server_{server_id}_messages"
        try:
            collection = self.db.get_collection(collection_name)
        except Exception:
            collection = self.db.create_collection(collection_name)
        return collection

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

            index = await asyncio.wait_for(
                asyncio.to_thread(
                    VectorStoreIndex.from_documents,
                    documents,
                    storage_context=storage_context,
                    show_progress=False,
                ),
                timeout=INDEX_TIMEOUT
            )
            return index
        except TimeoutError:
            print(f"Таймаут {INDEX_TIMEOUT} сек. при индексации сообщений сервера {server_id}")
            return None
        except Exception as e:
            print(f"Ошибка индексации сообщений: {e}")
            return None

    async def query_relevant_context(self, server_id: int, query: str, limit: int = 8) -> list[str]:
        """Найти релевантный контекст для запроса на сервере."""
        try:
            collection = self.get_server_collection(server_id)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = await asyncio.wait_for(
                asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store),
                timeout=QUERY_TIMEOUT
            )

            retriever = index.as_retriever(similarity_top_k=limit)
            nodes = await asyncio.wait_for(
                asyncio.to_thread(retriever.retrieve, query),
                timeout=QUERY_TIMEOUT
            )

            relevant_contexts = [node.text for node in nodes]
            return relevant_contexts
        except TimeoutError:
            print(f"Таймаут {QUERY_TIMEOUT} сек. при поиске контекста для сервера {server_id}")
            return []
        except Exception as e:
            print(f"Ошибка поиска контекста: {e}")
            return []

    async def index_server_users(self, server_id: int, users: list[str]) -> Any:
        """Индексировать список пользователей сервера с использованием метаданных."""
        try:
            collection = self.get_server_collection(server_id)
            await asyncio.wait_for(
                asyncio.to_thread(collection.delete, where={"document_type": "server_users"}),
                timeout=5.0
            )
            users_text = f"Список пользователей сервера: {', '.join(users)}"
            document = Document(
                text=users_text, metadata={"document_type": "server_users", "server_id": server_id}
            )

            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = await asyncio.wait_for(
                asyncio.to_thread(
                    VectorStoreIndex.from_documents,
                    [document],
                    storage_context=storage_context,
                    show_progress=False,
                ),
                timeout=INDEX_TIMEOUT
            )

            print(f"Обновлен список пользователей сервера {server_id}: {len(users)} пользователей")
            return index
        except TimeoutError:
            print(f"Таймаут при индексации пользователей сервера {server_id}")
            return None
        except Exception as e:
            print(f"Ошибка индексации пользователей сервера: {e}")
            return None
