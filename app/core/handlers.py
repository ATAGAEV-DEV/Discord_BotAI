import asyncio

import discord
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.core.ai_config import get_client, get_model
from app.core.config import AI_GENERATE_TIMEOUT
from app.data import user_descriptions_cache
from app.services.llama_integration import LlamaIndexManager
from app.tools.utils import (
    count_tokens,
    enrich_users_context,
    replace_emojis,
    resolve_user_mentions,
    strip_emoji,
    user_prompt,
)

llama_manager = LlamaIndexManager()


async def clear_server_history(server_id: int) -> str | None:
    """Очищает историю сообщений сервера в индексе LlamaIndex.

    Оставляет только документы типа 'server_users'.
    Дополнительно сбрасывает краткосрочную in-memory историю всех каналов сервера.
    """
    collection = llama_manager.get_server_collection(server_id)
    results = collection.get()

    # Сброс in-memory истории каналов сервера
    cleared_channels = llama_manager.clear_server_channel_histories(server_id)

    if results and "ids" in results and results["ids"]:
        ids_to_delete = []
        metadatas = results.get("metadatas", [])

        for i, metadata in enumerate(metadatas):
            if metadata.get("document_type") != "server_users":
                ids_to_delete.append(results["ids"][i])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            suffix = f", сброшена история {cleared_channels} канал(ов)" if cleared_channels else ""
            return f"Удалено {len(ids_to_delete)} документов из индекса сервера {server_id}{suffix}"
        else:
            suffix = f", сброшена история {cleared_channels} канал(ов)" if cleared_channels else ""
            return (
                f"В индексе сервера {server_id} нет документов для удаления "
                f"(кроме списка пользователей){suffix}"
            )
    else:
        return f"Индекс сервера {server_id} уже пуст"


async def ai_generate(
    text: str,
    server_id: int | None,
    name: str,
    limit: int = 15,
    emoji_ids: dict[str, str] | None = None,
    *,
    mentions: dict[int, str] | None = None,
    channel_id: int | None = None,
) -> str:
    """Генерирует ответ с контекстом упомянутых аккаунтов и глобальным таймаутом.

    ``channel_id`` используется для хранения краткосрочной памяти канала
    (последние 10 пар user/assistant). Если не передан — история не ведётся.
    """

    async def _generate_inner() -> str:
        """Внутренняя функция генерации (без таймаута)."""
        resolved_text, mentioned_names = resolve_user_mentions(text, mentions)

        # --- 1. Системный промпт ---
        system_msg: dict[str, str] = {
            "role": "system",
            "content": user_prompt(
                name,
                server_id,
                emoji_ids=emoji_ids,
                text=text,
                mentioned_names=mentioned_names,
            ),
        }

        # --- 2. Краткосрочная история канала ---
        history = (
            llama_manager.get_channel_history(channel_id, server_id)
            if channel_id is not None
            else None
        )
        history_messages = history.get_messages() if history is not None else []
        history_texts = history.get_texts() if history is not None else []

        # --- 3. RAG-контекст (без дублей из истории) ---
        relevant_contexts = await llama_manager.query_relevant_context(
            server_id, resolved_text, limit=limit, history_texts=history_texts
        )
        descriptions = user_descriptions_cache.get(server_id) if server_id is not None else {}
        relevant_contexts = enrich_users_context(relevant_contexts, descriptions)

        # --- 4. Сборка messages для API ---
        # Порядок: [system] → [преамбула+история] → [RAG] → [user]
        messages: list[dict[str, str]] = [system_msg]

        if history_messages:
            messages.append({
                "role": "system",
                "content": "Ниже — последние сообщения этого канала для контекста текущего разговора:",
            })
            messages.extend(history_messages)

        if relevant_contexts:
            messages.append({
                "role": "system",
                "content": (
                    "Релевантные фрагменты из долгосрочной памяти сервера"
                    " (архивные сообщения и профили участников):\n"
                    + "\n".join(relevant_contexts)
                ),
            })

        user_msg: dict[str, str] = {
            "role": "user",
            "content": f"[Пользователь: {name}] {resolved_text}",
        }
        messages.append(user_msg)

        # --- 5. Конвертация в типизированные OpenAI-объекты ---
        openai_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                openai_messages.append(
                    ChatCompletionSystemMessageParam(role="system", content=msg["content"])
                )
            elif role == "user":
                openai_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=msg["content"])
                )
            elif role == "assistant":
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"])
                )

        # --- 6. Вызов API ---
        completion = await get_client(server_id).chat.completions.create(
            model=get_model(server_id),
            messages=openai_messages,
            temperature=0.8,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            max_tokens=4500,
            timeout=AI_GENERATE_TIMEOUT,
        )

        response_text = completion.choices[0].message.content
        cleaned_response_text = discord.utils.remove_markdown(response_text)
        cleaned_response_text = replace_emojis(cleaned_response_text, emoji_ids or {})

        assistant_msg: dict[str, str] = {
            "role": "assistant",
            "content": strip_emoji(cleaned_response_text),
        }

        # --- 7. Сохраняем пару в краткосрочную историю ---
        if history is not None:
            history.add_pair(user_msg, assistant_msg)

        # --- 8. Индексируем пару в ChromaDB (долгосрочная память) ---
        asyncio.create_task(llama_manager.index_messages(server_id, [user_msg, assistant_msg]))

        print(f"Сообщения {messages}")
        print(count_tokens(messages))
        print(f"Ответ от ИИ: {response_text}")
        return cleaned_response_text

    try:
        return await asyncio.wait_for(_generate_inner(), timeout=AI_GENERATE_TIMEOUT)
    except TimeoutError:
        print(f"Таймаут {AI_GENERATE_TIMEOUT} сек. при генерации ответа для пользователя {name}")
        return f"⏳ Запрос не обработан за {AI_GENERATE_TIMEOUT} секунд. Попробуйте позже."
