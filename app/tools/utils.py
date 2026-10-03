import re

import discord
import tiktoken

from app.data import emoji_descriptions_cache, user_descriptions_cache
from app.tools.prompt import EMOJI_PROMPT, RANK_CONFIG, SYSTEM_PROMPT

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

_EMOJI_TAG_RE = re.compile(r"\[e:(\w+)\]")
_MALFORMED_EMOJI_RE = re.compile(
    r"(?P<protected>(?i:https?://|www\.)\S+|<a?:\w+:\d+>)"
    r"|(?<![\w:/\\<>\[\]])"
    r"(?::?e:(?P<prefixed_name>\w+)|:(?P<shortcode_name>\w+):)"
    r"(?![\w:/\\<>\[\]])"
)


def replace_emojis(text: str, emoji_ids: dict[str, str]) -> str:
    """Заменяет теги [e:name] и известные ошибочные маркеры на <:name:id>.

    Формы :e:name, :name: и e:name исправляются только при точном совпадении
    имени с emoji_ids и наличии ID. Неизвестные ошибочные маркеры, фрагменты
    слов, URL и готовых Discord-эмодзи не нормализуются.
    Правильный тег [e:name] без ID удаляется, как и раньше.
    """

    def _normalize(match: re.Match[str]) -> str:
        """Нормализует только отдельные ошибочные маркеры из серверного кэша."""
        if match.group("protected") is not None:
            return match.group(0)
        name = match.group("prefixed_name") or match.group("shortcode_name")
        if name and emoji_ids.get(name):
            return f"[e:{name}]"
        return match.group(0)

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        emoji_id = emoji_ids.get(name)
        if emoji_id:
            return f"<:{name}:{emoji_id}>"
        return ""

    normalized_text = _MALFORMED_EMOJI_RE.sub(_normalize, text)
    return _EMOJI_TAG_RE.sub(_replace, normalized_text)


def user_prompt(
    name: str,
    guild_id: int | None = None,
    emoji_ids: dict[str, str] | None = None,
) -> str:
    """Формирует системный prompt с данными текущего сервера.

    Непустое описание автора добавляется при точном совпадении имени на этом сервере,
    независимо от истории и эмодзи. Без ``guild_id`` описания пользователей не добавляются.
    В список эмодзи попадают только описания этого сервера с непустым ID в ``emoji_ids``.
    """
    descriptions = user_descriptions_cache.get(guild_id) if guild_id is not None else {}
    emoji_descriptions = emoji_descriptions_cache.get(guild_id) if guild_id is not None else {}

    emoji_ids = emoji_ids or {}
    emoji_list = "\n".join(
        f"[e:{emoji_name}] — {description}"
        for emoji_name, description in emoji_descriptions.items()
        if emoji_ids.get(emoji_name)
    )
    emoji_section = EMOJI_PROMPT.format(emoji_list=emoji_list).strip() + "\n" if emoji_list else ""

    user_info = ""
    description = descriptions.get(name, "")
    if description.strip():
        user_info = (
            "Информация по пользователям с name (они должны совпадать побуквенно, "
            "иначе это другой юзер). Но не упоминать об этом постоянно:"
            f"\n- {name}: {description}"
        )

    return SYSTEM_PROMPT.format(emoji_section=emoji_section, user_info=user_info).strip()


def enrich_users_context(contexts: list[str], user_descriptions: dict[str, str]) -> list[str]:
    """Обогащает контекст непустыми описаниями пользователей текущего сервера."""
    new_contexts = []

    for context in contexts:
        if context.startswith("Список пользователей сервера:"):
            users_str = context.replace("Список пользователей сервера:", "").strip()
            users_list = [user.strip() for user in users_str.split(",")]

            enriched_users = []
            for user in users_list:
                description = user_descriptions.get(user, "")
                if description.strip():
                    enriched_users.append(f"{user}: {description}")
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
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002600-\U000027bf"  # misc symbols
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U00002700-\U000027bf"  # dingbats
        "\U0001fa00-\U0001fa6f"  # chess, etc.
        "\U0001fa70-\U0001faff"  # food, etc.
        "\U00002500-\U00002bef"  # box drawing, arrows
        "\U0000fe00-\U0000fe0f"  # variation selectors
        "\U0001f004-\U0001f0cf"  # mahjong, playing cards
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
