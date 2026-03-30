import re
from datetime import datetime

import discord
import tiktoken

from app.tools.prompt import EMOJIS, RANK_CONFIG, SYSTEM_PROMPT, USER_DESCRIPTIONS

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")


def user_prompt(name: str) -> str:
    """Формирует системный промпт для пользователя.

    Если имя совпадает с ключами в P.USER_DESCRIPTIONS.
    """
    if str(name).strip() in USER_DESCRIPTIONS:
        user_info = (
            "Информация по пользователям с name (они должны совпадать побуквенно, "
            "иначе это другой юзер). Но не упоминать об этом постоянно:"
        )
        user_info += f"\n- {name}: {USER_DESCRIPTIONS[name]}"
        prompt = SYSTEM_PROMPT.format(user_info=user_info).strip()
        return prompt
    else:
        cleaned_prompt = re.sub(r"\n\s*5\..*", "", SYSTEM_PROMPT.strip())
        return cleaned_prompt


def enrich_users_context(contexts: list[str], user_descriptions: dict) -> list[str]:
    """Обогащает контекст информацией о пользователях из USER_DESCRIPTIONS."""
    new_contexts = []

    for context in contexts:
        if context.startswith("Список пользователей сервера:"):
            users_str = context.replace("Список пользователей сервера:", "").strip()
            users_list = [user.strip() for user in users_str.split(",")]

            enriched_users = []
            for user in users_list:
                if user in user_descriptions:
                    enriched_users.append(f"{user}: {user_descriptions[user]}")
                else:
                    enriched_users.append(user)

            new_context = "Список пользователей сервера: " + "; ".join(enriched_users)
            new_contexts.append(new_context)
        else:
            new_contexts.append(context)

    return new_contexts


def users_context(contexts: list[str], user_descriptions: dict) -> str:
    """Обогащает контекст информацией о пользователях из USER_DESCRIPTIONS."""
    users_list = [user.strip() for user in contexts]

    enriched_users = []
    for user in users_list:
        if user in user_descriptions:
            enriched_users.append(f"{user}: {user_descriptions[user]}")

    new_context = "Список пользователей сервера: " + "; ".join(enriched_users)
    return new_context


def contains_only_urls(text: str) -> bool:
    """Проверяет, содержит ли текст только ссылки (и пробелы между ними)."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    text_without_urls = url_pattern.sub("", text)
    return not text_without_urls.strip()


def darken_color(rgb: tuple, factor: float = 0.75) -> tuple:
    """Уменьшает яркость цвета RGB — делает его темнее.

    factor < 1 = темнее, factor > 1 = светлее.
    """
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def count_tokens(text: str | None) -> int:
    """Подсчитывает количество токенов в тексте с использованием кодировки GPT-4o-mini."""
    if not isinstance(text, str):
        text = str(text) if text else ""
    return len(ENCODING.encode(text))


def clean_text(text: str) -> str:
    """Очищает текст от markdown-стилей: **, *, ###, ##, #."""
    cleaned_text = re.sub(r"(\*\*|\*|###|##|#)", "", text)
    return cleaned_text


def replace_emojis(text: str) -> str:
    """Заменяет текстовые представления эмодзи на реальные Discord-эмодзи."""
    for emoji in EMOJIS:
        pattern = re.compile(rf":?\s*{re.escape(emoji.slug)}\s*:?")
        text = pattern.sub(emoji.full_code, text)

    f_emoji_code = next((e.full_code for e in EMOJIS if e.slug == "F_"), "")
    if f_emoji_code:
        pattern_f = re.compile(r":\s*F\s*:?")
        text = pattern_f.sub(f_emoji_code, text)

    return text


COLOR_MAP: dict[str, discord.Color] = {
    "light_grey": discord.Color.light_grey(),
    "green": discord.Color.green(),
    "blue": discord.Color.blue(),
    "red": discord.Color.red(),
    "purple": discord.Color.purple(),
    "gold": discord.Color.gold(),
}


def get_rank_description(message_count: int) -> dict:
    """Возвращает описание уровня (ранга) пользователя на основе количества сообщений."""
    selected = RANK_CONFIG[0]
    for rank_cfg in RANK_CONFIG:
        if message_count >= rank_cfg["threshold"]:
            selected = rank_cfg

    return {
        "color": COLOR_MAP.get(selected["color_name"], discord.Color.default()),
        "next_threshold": selected["next_threshold"],
        "rank_level": RANK_CONFIG.index(selected),
        "text_color": selected["text_color"],
        "bg_filename": selected["bg_filename"],
        "description": selected["name"],
    }


def convert_mcp_tools_to_openai(mcp_tools: list) -> list:
    """Конвертирует MCP инструменты в формат OpenAI."""
    openai_tools = []

    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "Инструмент без описания",
                "parameters": tool.inputSchema,
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def parse_birthday_date(content: str) -> datetime:
    """Парсит дату рождения из текста команды.

    Ожидаемый формат: '!birthday DD.MM.YYYY' или просто 'DD.MM.YYYY'.
    """
    # Убираем возможный префикс команды
    text = content.strip()
    for prefix in ("!birthday", "?birthday"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    if not re.match(r"^\d{2}\.\d{2}\.\d{4}$", text):
        raise ValueError("Некорректный формат даты. Используйте DD.MM.YYYY.")

    try:
        return datetime.strptime(text, "%d.%m.%Y")
    except Exception:
        raise ValueError("Некорректный формат даты. Используйте DD.MM.YYYY.")


def parse_holiday_command(content: str) -> tuple[int, int, str]:
    """Парсит команду добавления праздника.

    Ожидаемый формат: '!holiday DD.MM Название праздника'.
    Возвращает (day, month, holiday_name).
    """
    try:
        args = content.split(" ", 2)
        if len(args) < 3:
            raise ValueError("Используйте формат: `!holiday DD.MM Название праздника`")

        date_str = args[1]
        holiday_name = args[2]

        if not re.match(r"^\d{2}\.\d{2}$", date_str):
            raise ValueError("Некорректный формат даты. Используйте DD.MM (например, 01.01).")

        day, month = map(int, date_str.split("."))

        if not (1 <= month <= 12):
            raise ValueError("Месяц должен быть от 1 до 12.")

        if not (1 <= day <= 31):
            raise ValueError("День должен быть от 1 до 31.")

        return day, month, holiday_name

    except ValueError as ve:
        raise ve
    except Exception:
        raise ValueError("Ошибка разбора команды. Используйте DD.MM Название.")


def chunk_message(text: str, limit: int = 1900) -> list[str]:
    """Разбивает длинное сообщение на части не более limit символов.

    Разделение происходит по строкам, чтобы не разрывать слова.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.split("\n"):
        # Если одна строка длиннее лимита — разбиваем принудительно
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]

        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
