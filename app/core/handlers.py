import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.core.ai_config import get_client, get_mini_model, get_model
from app.services.llama_integration import LlamaIndexManager
from app.tools.prompt import SEARCH_PROMPT, SYSTEM_BIRTHDAY_PROMPT, USER_DESCRIPTIONS, WEATHER_PROMPT
from app.tools.utils import (
    clean_text,
    convert_mcp_tools_to_openai,
    count_tokens,
    enrich_users_context,
    replace_emojis,
    user_prompt,
)

llama_manager = LlamaIndexManager()


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
    tool_weather: str | None,
    tool_search: str | None,
    limit: int = 15,
) -> str:
    """Генерирует ответ от AI на основе контекста сервера и текущего сообщения пользователя."""
    messages = [{"role": "system", "content": user_prompt(f"{name}")}]
    relevant_contexts = await llama_manager.query_relevant_context(server_id, text, limit=limit)
    relevant_contexts = enrich_users_context(relevant_contexts, USER_DESCRIPTIONS)

    if relevant_contexts:
        context_message = {
            "role": "system",
            "content": "Релевантный контекст из истории сервера:\n" + "\n".join(relevant_contexts),
        }
        messages.append(context_message)

    if tool_weather is not None:
        tool_message = {
            "role": "system",
            "content": f"Дополнительная информация от инструментов: {tool_weather}",
        }
        messages.append(tool_message)

    if tool_search is not None:
        tool_message = {
            "role": "system",
            "content": f"Дополнительная информация от инструментов: {tool_search}",
        }
        messages.append(tool_message)

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
        emoji_response_text = replace_emojis(cleaned_response_text)

        messages_to_index = [
            {"role": "user", "content": f"[Пользователь: {name}] {text}"},
            {"role": "assistant", "content": cleaned_response_text},
        ]
        await llama_manager.index_messages(server_id, messages_to_index)
        # print(f"Релевантный {relevant_contexts}")
        # print(count_tokens(relevant_contexts))
        print(f"Сообщения {messages}")
        print(count_tokens(messages))
        print(f"Ответ с эмодзи: {emoji_response_text}")
        print(f"Ответ от ИИ: {response_text}")
        return emoji_response_text
    except Exception as e:
        print(f"Ошибка при вызове OpenAI API: {e}")
        return "Произошла ошибка. Пожалуйста, попробуйте позже."


async def ai_generate_birthday_congrats(name: str) -> str:
    """Генерирует креативное поздравление с днём рождения для пользователя."""
    prompt = [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_BIRTHDAY_PROMPT.strip()),
        ChatCompletionUserMessageParam(
            role="user", content=f"Сгенерируй креативное поздравление с днем рождения для {name}."
        ),
    ]

    try:
        completion = await get_client().chat.completions.create(
            model=get_model(),
            messages=prompt,
            temperature=0.8,  # Оптимальный баланс креативности/когерентности
            top_p=0.8,  # Шире выборка слов
            frequency_penalty=0.1,  # Поощряет новые формулировки
            presence_penalty=0.2,  # Поощряет новые темы
            max_tokens=400,
            timeout=30.0,
        )
        text = completion.choices[0].message.content.strip()
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"[Ошибка генерации поздравления]: {e}")
        return "Поздравляем с днём рождения! 🎉"


async def check_tool_intent(text: str, server_script: str, prompt: str) -> str | None:
    """Проверяет наличие намерения использовать MCP-инструмент и обрабатывает его.

    Универсальная функция для проверки намерений (погода, поиск и т.д.).
    """
    server_params = StdioServerParameters(command="python", args=[server_script], env=None)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_list = await session.list_tools()

            openai_tools = convert_mcp_tools_to_openai(tools_list.tools)

            messages = [
                ChatCompletionSystemMessageParam(role="system", content=prompt.strip()),
                ChatCompletionUserMessageParam(role="user", content=text),
            ]

            final_response = await process_mcp_conversation(messages, openai_tools, session)

            if final_response[1] is True:
                return final_response[0]
            return None


async def check_weather_intent(text: str) -> str | None:
    """Проверяет наличие намерения запросить информацию о погоде."""
    return await check_tool_intent(text, "app/mcp/server_weather.py", WEATHER_PROMPT)


async def check_search_intent(text: str) -> str | None:
    """Проверяет наличие намерения запросить информацию через поиск."""
    return await check_tool_intent(text, "app/mcp/server_search.py", SEARCH_PROMPT)


async def process_mcp_conversation(
    messages: list, tools: list, session: ClientSession
) -> tuple[str, bool]:
    """Обрабатывает разговор с возможными вызовами MCP-инструментов."""
    try:
        response = await get_client().chat.completions.create(
            model=get_mini_model(),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=30.0,
        )
    except Exception as e:
        print(f"Ошибка при вызове ИИ провайдера (MCP_conversation): {e}")
        return "", False
    assistant_message = response.choices[0].message
    print(assistant_message)

    if not assistant_message.tool_calls:
        return assistant_message.content, False

    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        try:
            result = await session.call_tool(function_name, function_args)

            if result.content:
                tool_result = result.content[0].text if result.content else "Нет результата"
            else:
                tool_result = "Инструмент выполнен, но результат пуст"

            return tool_result, True

        except Exception as e:
            print(f"Ошибка при вызове инструмента: {str(e)}")

    return "", False
