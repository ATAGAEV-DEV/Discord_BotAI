import re

import discord
import tiktoken

from app.data import emoji_descriptions_cache, user_descriptions_cache
from app.tools.prompt import EMOJI_LIST_STRING, EMOJIS, RANK_CONFIG, SYSTEM_PROMPT

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

_EMOJI_TAG_RE = re.compile(r"\[e:(\w+)\]")



def replace_emojis(text: str, emoji_ids: dict[str, str]) -> str:
    """Заменяет теги [e:name] на Discord-эмодзи <:name:id>.

    Если эмодзи с таким именем не найден в emoji_ids — тег удаляется.
    """
    def _replace(match: re.Match) -> str:
        name = match.group(1)
        emoji_id = emoji_ids.get(name)
        if emoji_id:
            return f"<:{name}:{emoji_id}>"
        return ""

    return _EMOJI_TAG_RE.sub(_replace, text)


def user_prompt(name: str, guild_id: int | None = None) -> str:
    """Формирует системный prompt с данными текущего сервера.

    ``guild_id`` необязателен для обратной совместимости со старыми вызовами.
    В рабочем Discord-потоке он передаётся всегда, чтобы данные разных серверов
    не смешивались.
    """
    descriptions = user_descriptions_cache.get_all()
    emoji_descriptions = (
        emoji_descriptions_cache.get(guild_id) if guild_id is not None else {}
    )

    # EMOJIS намеренно остаётся в prompt.py как fallback и источник для переноса
    # исходных описаний в БД.
    configured_emojis = {**EMOJIS, **emoji_descriptions}
    emoji_list = (
        "\n".join(
            f"[e:{emoji_name}] — {description}"
            for emoji_name, description in configured_emojis.items()
        )
        if configured_emojis
        else EMOJI_LIST_STRING
    )

    if str(name).strip() in descriptions:
        user_info = (
            "Информация по пользователям с name (они должны совпадать побуквенно, "
            "иначе это другой юзер). Но не упоминать об этом постоянно:"
            f"\n- {name}: {descriptions[name]}"
        )
        user_prompt_text = SYSTEM_PROMPT.format(
            user_info=user_info,
            emoji_list=emoji_list,
        ).strip()
        return user_prompt_text

    return re.sub(
        r"\n\s*5\..*",
        "",
        SYSTEM_PROMPT.format(user_info="", emoji_list=emoji_list).strip(),
    )


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


def strip_emoji(text: str) -> str:
    """Удаляет все Unicode-эмодзи и кастомные Discord-эмодзи из текста для индексации в ChromaDB.

    Используется перед сохранением в векторное хранилище, чтобы
    модель не обучалась на примерах ответов с эмодзи.
    """
    # Кастомные Discord-эмодзи: <:name:id> и <a:name:id> (анимированные)
    text = re.sub(r"<a?:\w+:\d+>", "", text)

    # Диапазоны Unicode-блоков эмодзи
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002600-\U000027BF"  # misc symbols
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001FA00-\U0001FA6F"  # chess, etc.
        "\U0001FA70-\U0001FAFF"  # food, etc.
        "\U00002500-\U00002BEF"  # box drawing, arrows
        "\U0000FE00-\U0000FE0F"  # variation selectors
        "\U0001F004-\U0001F0CF"  # mahjong, playing cards
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


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
