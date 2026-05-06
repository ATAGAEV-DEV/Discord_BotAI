"""Unit-тесты для app/tools/utils.py."""

from types import SimpleNamespace

import pytest

from app.tools.prompt import RANK_NAMES
from app.tools.utils import (
    clean_text,
    contains_only_urls,
    convert_mcp_tools_to_openai,
    count_tokens,
    darken_color,
    enrich_users_context,
    get_rank_description,
    replace_emojis,
    user_prompt,
    users_context,
)

# ── contains_only_urls ──────────────────────────────────────────


class TestContainsOnlyUrls:
    """Тесты для функции contains_only_urls."""

    def test_single_url(self) -> None:
        """Одна ссылка — True."""
        assert contains_only_urls("https://example.com") is True

    def test_multiple_urls(self) -> None:
        """Несколько ссылок через пробел — True."""
        assert contains_only_urls("https://a.com https://b.com") is True

    def test_www_url(self) -> None:
        """www-ссылка — True."""
        assert contains_only_urls("www.example.com") is True

    def test_text_with_url(self) -> None:
        """Текст + ссылка — False."""
        assert contains_only_urls("смотри https://example.com") is False

    def test_plain_text(self) -> None:
        """Обычный текст без ссылок — False."""
        assert contains_only_urls("Привет мир") is False

    def test_empty_string(self) -> None:
        """Пустая строка — True (нет не-URL символов)."""
        assert contains_only_urls("") is True

    def test_whitespace_only(self) -> None:
        """Только пробелы — True."""
        assert contains_only_urls("   ") is True


# ── darken_color ────────────────────────────────────────────────


class TestDarkenColor:
    """Тесты для функции darken_color."""

    def test_default_factor(self) -> None:
        """Стандартный коэффициент 0.75."""
        result = darken_color((200, 100, 50))
        assert result == (150, 75, 37)

    def test_custom_factor(self) -> None:
        """Пользовательский коэффициент 0.5."""
        result = darken_color((200, 100, 50), factor=0.5)
        assert result == (100, 50, 25)

    def test_factor_greater_than_1(self) -> None:
        """Коэффициент > 1 — осветление."""
        result = darken_color((100, 100, 100), factor=1.5)
        assert result == (150, 150, 150)

    def test_clamp_to_zero(self) -> None:
        """Не уходит ниже 0."""
        result = darken_color((0, 0, 0), factor=0.5)
        assert result == (0, 0, 0)

    def test_clamp_to_255(self) -> None:
        """Не уходит выше 255."""
        result = darken_color((200, 200, 200), factor=2.0)
        assert result == (255, 255, 255)


# ── count_tokens ────────────────────────────────────────────────


class TestCountTokens:
    """Тесты для функции count_tokens."""

    def test_normal_text(self) -> None:
        """Обычный текст возвращает > 0 токенов."""
        assert count_tokens("Hello, world!") > 0

    def test_empty_string(self) -> None:
        """Пустая строка → 0 токенов."""
        assert count_tokens("") == 0

    def test_none_value(self) -> None:
        """None → не падает, возвращает > 0."""
        result = count_tokens(None)
        assert result >= 0

    def test_returns_int(self) -> None:
        """Проверяем тип результата."""
        assert isinstance(count_tokens("тест"), int)


# ── get_rank_description ────────────────────────────────────────


class TestGetRankDescription:
    """Тесты для функции get_rank_description."""

    def test_zero_messages(self) -> None:
        """0 сообщений — 'Человек'."""
        rank = get_rank_description(0)
        assert rank["description"] == RANK_NAMES[0]
        assert rank["rank_level"] == 0

    def test_1_message(self) -> None:
        """1 сообщение — 'Начинающий бич'."""
        rank = get_rank_description(1)
        assert rank["description"] == RANK_NAMES[1]
        assert rank["rank_level"] == 1

    def test_49_messages(self) -> None:
        """49 сообщений — всё ещё 'Начинающий бич'."""
        rank = get_rank_description(49)
        assert rank["rank_level"] == 1

    def test_50_messages(self) -> None:
        """50 сообщений — 'Радужный бич'."""
        rank = get_rank_description(50)
        assert rank["description"] == RANK_NAMES[2]
        assert rank["rank_level"] == 2

    def test_100_messages(self) -> None:
        """100 сообщений — 'Бич'."""
        rank = get_rank_description(100)
        assert rank["description"] == RANK_NAMES[3]
        assert rank["rank_level"] == 3

    def test_200_messages(self) -> None:
        """200 сообщений — 'Босс бичей'."""
        rank = get_rank_description(200)
        assert rank["description"] == RANK_NAMES[4]
        assert rank["rank_level"] == 4

    def test_500_messages(self) -> None:
        """500 сообщений — 'Бич-император'."""
        rank = get_rank_description(500)
        assert rank["description"] == RANK_NAMES[5]
        assert rank["rank_level"] == 5

    def test_1000_messages(self) -> None:
        """1000+ — всё ещё 'Бич-император'."""
        rank = get_rank_description(1000)
        assert rank["rank_level"] == 5

    def test_rank_has_all_keys(self) -> None:
        """У каждого ранга есть все нужные ключи."""
        for count in [0, 1, 50, 100, 200, 500]:
            rank = get_rank_description(count)
            assert "color" in rank
            assert "next_threshold" in rank
            assert "rank_level" in rank
            assert "text_color" in rank
            assert "bg_filename" in rank
            assert "description" in rank


# ── user_prompt ─────────────────────────────────────────────────


class TestUserPrompt:
    """Тесты для функции user_prompt."""

    def test_known_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Известный пользователь — промпт содержит описание."""
        from app.data import user_descriptions_cache

        monkeypatch.setattr(
            user_descriptions_cache, "_cache", {0: {"atagaev": "Арби, создатель бота"}}
        )
        result = user_prompt("atagaev")
        assert "atagaev" in result
        assert "Арби" in result

    def test_unknown_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Неизвестный пользователь — промпт без user_info."""
        from app.data import user_descriptions_cache

        monkeypatch.setattr(user_descriptions_cache, "_cache", {})
        result = user_prompt("random_user_12345")
        assert "random_user_12345" not in result

    def test_returns_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Всегда возвращает строку."""
        from app.data import user_descriptions_cache

        monkeypatch.setattr(
            user_descriptions_cache, "_cache", {0: {"atagaev": "Арби, создатель бота"}}
        )
        assert isinstance(user_prompt("atagaev"), str)
        assert isinstance(user_prompt("unknown"), str)


# ── enrich_users_context ────────────────────────────────────────


class TestEnrichUsersContext:
    """Тесты для функции enrich_users_context."""

    def test_enriches_known_users(self) -> None:
        """Обогащает контекст описаниями известных пользователей."""
        contexts = ["Список пользователей сервера: atagaev, walkmantm"]
        descriptions = {"atagaev": "Арби", "walkmantm": "Кирилл"}
        result = enrich_users_context(contexts, descriptions)
        assert "atagaev: Арби" in result[0]
        assert "walkmantm: Кирилл" in result[0]

    def test_unknown_users_kept(self) -> None:
        """Неизвестные пользователи сохраняются как есть."""
        contexts = ["Список пользователей сервера: unknown_user"]
        result = enrich_users_context(contexts, {"atagaev": "Арби"})
        assert "unknown_user" in result[0]

    def test_non_user_context_unchanged(self) -> None:
        """Не-пользовательские контексты не изменяются."""
        contexts = ["Какой-то другой контекст"]
        result = enrich_users_context(contexts, {"atagaev": "Арби"})
        assert result[0] == "Какой-то другой контекст"

    def test_empty_contexts(self) -> None:
        """Пустой список — пустой результат."""
        assert enrich_users_context([], {}) == []


# ── users_context ───────────────────────────────────────────────


class TestUsersContext:
    """Тесты для функции users_context."""

    def test_with_descriptions(self) -> None:
        """Формирует контекст с описаниями."""
        result = users_context(["atagaev"], {"atagaev": "Арби"})
        assert "atagaev: Арби" in result
        assert "Список пользователей сервера:" in result

    def test_without_descriptions(self) -> None:
        """Неизвестные пользователи не включаются."""
        result = users_context(["unknown"], {"atagaev": "Арби"})
        assert "unknown" not in result


# ── convert_mcp_tools_to_openai ─────────────────────────────────


class TestConvertMcpToolsToOpenai:
    """Тесты для функции convert_mcp_tools_to_openai."""

    def test_converts_tool(self) -> None:
        """Корректно конвертирует MCP-инструмент в формат OpenAI."""
        mock_tool = SimpleNamespace(
            name="test_tool",
            description="Тестовый инструмент",
            inputSchema={"type": "object", "properties": {}},
        )
        result = convert_mcp_tools_to_openai([mock_tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
        assert result[0]["function"]["description"] == "Тестовый инструмент"

    def test_no_description_fallback(self) -> None:
        """Без описания — fallback-текст."""
        mock_tool = SimpleNamespace(
            name="tool",
            description=None,
            inputSchema={},
        )
        result = convert_mcp_tools_to_openai([mock_tool])
        assert result[0]["function"]["description"] == "Инструмент без описания"

    def test_empty_list(self) -> None:
        """Пустой список → пустой результат."""
        assert convert_mcp_tools_to_openai([]) == []


# ── clean_text ──────────────────────────────────────────────────


class TestCleanText:
    """Тесты для функции clean_text."""

    def test_removes_bold(self) -> None:
        """Удаляет **жирный** markdown."""
        result = clean_text("**Жирный** текст")
        assert "**" not in result
        assert "Жирный" in result

    def test_removes_headers(self) -> None:
        """Удаляет заголовки #, ##, ###."""
        result = clean_text("### Заголовок")
        assert "#" not in result
        assert "Заголовок" in result

    def test_plain_text_unchanged(self) -> None:
        """Обычный текст не меняется."""
        result = clean_text("обычный текст")
        assert result == "обычный текст"


# ── replace_emojis ──────────────────────────────────────────────


class TestReplaceEmojis:
    """Тесты для функции replace_emojis."""

    def test_replaces_known_emoji(self) -> None:
        """Заменяет известные текстовые эмодзи на Discord-формат."""
        result = replace_emojis("Привет :yoba: мир")
        assert "<:yoba:1101900451852599427>" in result
        assert "Привет" in result

    def test_no_emoji_unchanged(self) -> None:
        """Текст без эмодзи не меняется."""
        result = replace_emojis("обычный текст")
        assert result == "обычный текст"
