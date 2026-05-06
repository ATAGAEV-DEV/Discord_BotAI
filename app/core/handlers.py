import asyncio

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.core.ai_config import get_client, get_model
from app.data import user_descriptions_cache
from app.services.llama_integration import LlamaIndexManager
from app.tools.utils import (
    clean_text,
    count_tokens,
    enrich_users_context,
    user_prompt,
)

llama_manager = LlamaIndexManager()

AI_GENERATE_TIMEOUT = 100.0


async def clear_server_history(server_id: int) -> str | None:
    """Очищает историю сообщений сервера в индексе LlamaIndex.

    Оставляет только документы типа 'server_users'.
    """
    try:
        collection = llama_manager.get_server_collection(server_id)
        results = collection.get()
        if results and "ids" in results and results["ids"]:
            ids_to_delete = []
            metadatas = results.get("metadatas", [])

            for i, metadata in enumerate(metadatas):
                if metadata.get("document_type") != "server_users":
                    ids_to_delete.append(results["ids"][i])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                return f"Удалено {len(ids_to_delete)} документов из индекса сервера {server_id}"
            else:
                return (
                    f"В индексе сервера {server_id} нет документов для удаления "
                    "(кроме списка пользователей)"
                )
        else:
            return f"Индекс сервера {server_id} уже пуст"
    except Exception as e:
        print(f"Ошибка очистки индекса LlamaIndex: {e}")
        return f"Произошла ошибка при очистке индекса: {e}"


async def ai_generate(
    text: str,
    server_id: int,
    name: str,
    limit: int = 15,
) -> str:
    """Генерирует ответ от AI с глобальным таймаутом."""

    async def _generate_inner() -> str:
        """Внутренняя функция генерации (без таймаута)."""
        messages = [{"role": "system", "content": user_prompt(f"{name}")}]
        relevant_contexts = await llama_manager.query_relevant_context(server_id, text, limit=limit)
        descriptions = user_descriptions_cache.get_all()
        relevant_contexts = enrich_users_context(relevant_contexts, descriptions)

        if relevant_contexts:
            context_message = {
                "role": "system",
                "content": (
                    "Релевантный контекст из истории сервера:\n" + "\n".join(relevant_contexts)
                ),
            }
            messages.append(context_message)

        user_msg = {"role": "user", "content": f"[Пользователь: {name}] {text}"}
        messages.append(user_msg)

        try:
            openai_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    openai_messages.append(
                        ChatCompletionSystemMessageParam(role="system", content=msg["content"])
                    )
                elif msg["role"] == "user":
                    openai_messages.append(
                        ChatCompletionUserMessageParam(role="user", content=msg["content"])
                    )

            completion = await get_client().chat.completions.create(
                model=get_model(),
                messages=openai_messages,
                temperature=0.8,
                top_p=0.8,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                max_tokens=4500,
                timeout=60.0,
            )

            response_text = completion.choices[0].message.content
            cleaned_response_text = clean_text(response_text)

            messages_to_index = [
                {"role": "user", "content": f"[Пользователь: {name}] {text}"},
                {"role": "assistant", "content": cleaned_response_text},
            ]
            # Индексация в фоновом режиме (не блокирует ответ)
            asyncio.create_task(llama_manager.index_messages(server_id, messages_to_index))
            print(f"Сообщения {messages}")
            print(count_tokens(messages))
            print(f"Ответ от ИИ: {response_text}")
            return cleaned_response_text
        except Exception as e:
            print(f"Ошибка при вызове OpenAI API: {e}")
            return "Произошла ошибка. Пожалуйста, попробуйте позже."

    try:
        return await asyncio.wait_for(_generate_inner(), timeout=AI_GENERATE_TIMEOUT)
    except TimeoutError:
        print(f"Таймаут {AI_GENERATE_TIMEOUT} сек. при генерации ответа для пользователя {name}")
        return f"⏳ Запрос не обработан за {AI_GENERATE_TIMEOUT} секунд. Попробуйте позже."
