from datetime import datetime, timedelta

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from app.core.ai_config import get_client, get_model
from app.data import user_descriptions_cache
from app.tools.prompt import system_holiday_prompt
from app.tools.utils import clean_text, replace_emojis, users_context


async def ai_generate_holiday_congrats(names: list[str], holiday: str) -> str:
    """Генерирует креативное поздравление с праздником для пользователей."""
    relevant_contexts = users_context(names, user_descriptions_cache.get_all())
    current_date = datetime.now()
    date_minus_month = current_date - timedelta(days=30)
    date_plus_month = current_date + timedelta(days=30)
    old_year = date_minus_month.year
    new_year = date_plus_month.year
    if current_date.month == 1 and current_date.day == 1:
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=system_holiday_prompt(holiday).strip()
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"{relevant_contexts}. Новый год {new_year}. Старый год {old_year}.",
            ),
        ]
    elif holiday == "Днем Бичей":
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=system_holiday_prompt(holiday).strip()
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=(
                    f"{relevant_contexts}. Сегодня {current_date}, Сервер бичей открылся 2017 году."
                ),
            ),
        ]
    else:
        messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=system_holiday_prompt(holiday).strip()
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"{relevant_contexts}. Праздник: {holiday}."
            ),
        ]
    try:
        completion = await get_client().chat.completions.create(
            model=get_model(),
            messages=messages,
            temperature=1,  # Оптимальный баланс креативности/когерентности
            top_p=0.8,  # Шире выборка слов
            frequency_penalty=0.1,  # Поощряет новые формулировки
            presence_penalty=0.2,  # Поощряет новые темы
            max_tokens=5000,
        )
        text = completion.choices[0].message.content.strip()
        cleaned_response_text = clean_text(text)
        emoji_response_text = replace_emojis(cleaned_response_text)
        return emoji_response_text
    except Exception as e:
        print(f"[Ошибка генерации поздравления]: {e}")
        return f"Поздравляем всех пользователей с {holiday}!!!! 🎉"
