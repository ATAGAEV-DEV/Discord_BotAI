"""Unit-тесты для app/tools/utils.py."""

import pytest

from app.tools.prompt import RANK_NAMES
from app.tools.utils import (
    chunk_message,
    contains_only_urls,
    count_tokens,
    darken_color,
    enrich_users_context,
    get_rank_description,
    replace_emojis,
    strip_emoji,
    user_prompt,
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


# ── chunk_message ────────────────────────────────────────────────


class TestChunkMessage:
    """Тесты для функции chunk_message."""

    def test_short_message_is_unchanged(self) -> None:
        """Сообщение в пределах лимита не разбивается."""
        assert chunk_message("short", limit=10) == ["short"]

    def test_splits_by_lines(self) -> None:
        """Строки объединяются, пока результат не превышает лимит."""
        assert chunk_message("a\nb\nc", limit=3) == ["a\nb", "c"]

    def test_splits_overlong_line(self) -> None:
        """Одна слишком длинная строка разбивается принудительно."""
        assert chunk_message("12345", limit=3) == ["123", "45"]

    def test_flushes_current_chunk_before_overlong_line(self) -> None:
        """Накопленный текст отделяется перед длинной строкой."""
        assert chunk_message("a\n12345", limit=3) == ["a", "123", "45"]

    def test_does_not_append_empty_final_chunk(self) -> None:
        """Пустая строка в конце не добавляет пустой чанк."""
        assert chunk_message("a\n", limit=1) == ["a"]


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

    @pytest.mark.parametrize("name", ["atagaev", "unknown"], ids=["known-user", "unknown-user"])
    @pytest.mark.parametrize("guild_id", [None, 123], ids=["fallback", "guild-emojis"])
    def test_emoji_format_rules_survive_prompt_rendering(
        self, monkeypatch: pytest.MonkeyPatch, name: str, guild_id: int | None
    ) -> None:
        """Правила эмодзи сохраняются при подстановке списка и удалении пустого user_info."""
        from app.data import emoji_descriptions_cache, user_descriptions_cache

        monkeypatch.setattr(
            user_descriptions_cache, "_cache", {0: {"atagaev": "Арби, создатель бота"}}
        )
        monkeypatch.setattr(
            emoji_descriptions_cache,
            "_cache",
            {
                123: {"ServerLaugh42": "серверный смех"},
                456: {"ForeignEmoji": "чужой смех"},
            },
        )

        result = user_prompt(name, guild_id=guild_id)

        required_rules = (
            "4. Использовать только эмодзи сервера из списка ниже.",
            "Копируй тег из списка точно: [e:ИМЯ], где ИМЯ — имя выбранного эмодзи.",
            "Обязательно сохраняй обе квадратные скобки, префикс e: и регистр имени.",
            "Правильный пример: Ну и отлично [e:Gachi1]",
            "Неправильные варианты: :e:Gachi1, :Gachi1:, e:Gachi1.",
            "Не копируй неправильное написание эмодзи из истории сообщений.",
            "Перед выдачей ответа проверь, что каждый использованный тег "
            "точно совпадает с тегом из списка.",
            "Доступные эмодзи:",
        )
        for rule in required_rules:
            assert rule in result

        assert "{emoji_list}" not in result
        assert "{user_info}" not in result
        assert "[e:Gachi1] — смех" in result
        assert "[e:ForeignEmoji]" not in result

        if guild_id is None:
            assert "[e:ServerLaugh42]" not in result
        else:
            assert "[e:ServerLaugh42] — серверный смех" in result

        if name == "atagaev":
            assert "5. Информация по пользователям" in result
            assert "- atagaev: Арби, создатель бота" in result
        else:
            assert "5. " not in result
            assert "Арби" not in result

    def test_guild_emoji_description_overrides_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Описание эмодзи из БД сервера заменяет одноимённый fallback."""
        from app.data import emoji_descriptions_cache, user_descriptions_cache

        monkeypatch.setattr(user_descriptions_cache, "_cache", {})
        monkeypatch.setattr(
            emoji_descriptions_cache, "_cache", {123: {"yoba": "локальное описание"}}
        )

        result = user_prompt("unknown", guild_id=123)

        assert "[e:yoba] — локальное описание" in result
        assert "[e:Gachi1] — смех" in result

    def test_guild_emoji_descriptions_are_isolated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Описание эмодзи другого сервера не попадает в prompt."""
        from app.data import emoji_descriptions_cache, user_descriptions_cache

        monkeypatch.setattr(user_descriptions_cache, "_cache", {})
        monkeypatch.setattr(
            emoji_descriptions_cache,
            "_cache",
            {123: {"yoba": "сервер один"}, 456: {"yoba": "сервер два"}},
        )

        result = user_prompt("unknown", guild_id=123)

        assert "сервер один" in result
        assert "сервер два" not in result


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


# ── replace_emojis ──────────────────────────────────────────────


class TestReplaceEmojis:
    """Тесты для функции replace_emojis."""

    EMOJI_IDS = {"yoba": "1101900451852599427", "Gachi1": "469464559959277578"}
    MALFORMED_FORMATS = [":e:{name}", ":{name}:", "e:{name}"]

    def test_replaces_known_emoji(self) -> None:
        """Известный эмодзи заменяется на Discord-формат."""
        result = replace_emojis("привет [e:yoba] пока", self.EMOJI_IDS)
        assert result == "привет <:yoba:1101900451852599427> пока"

    def test_removes_unknown_emoji(self) -> None:
        """Неизвестный эмодзи удаляется из текста."""
        result = replace_emojis("текст [e:unknown] конец", self.EMOJI_IDS)
        assert "[e:unknown]" not in result
        assert "текст" in result
        assert "конец" in result

    def test_multiple_emojis(self) -> None:
        """Несколько эмодзи обрабатываются корректно."""
        result = replace_emojis("[e:yoba] смеюсь [e:Gachi1]", self.EMOJI_IDS)
        assert "<:yoba:1101900451852599427>" in result
        assert "<:Gachi1:469464559959277578>" in result

    def test_mixed_known_and_unknown(self) -> None:
        """Известный заменяется, неизвестный удаляется."""
        result = replace_emojis("[e:yoba] и [e:ghost]", self.EMOJI_IDS)
        assert "<:yoba:1101900451852599427>" in result
        assert "[e:ghost]" not in result

    def test_no_emoji_unchanged(self) -> None:
        """Текст без эмодзи-тегов не меняется."""
        result = replace_emojis("просто текст без тегов", self.EMOJI_IDS)
        assert result == "просто текст без тегов"

    def test_empty_string(self) -> None:
        """Пустая строка — пустая строка."""
        assert replace_emojis("", self.EMOJI_IDS) == ""

    def test_empty_emoji_ids(self) -> None:
        """Пустой кэш — все теги удаляются."""
        result = replace_emojis("текст [e:yoba] конец", {})
        assert "[e:yoba]" not in result
        assert "текст" in result

    def test_returns_string(self) -> None:
        """Всегда возвращает строку."""
        assert isinstance(replace_emojis("тест [e:yoba]", self.EMOJI_IDS), str)

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    @pytest.mark.parametrize("name", ["yoba", "Gachi1", "ServerLaugh42"])
    def test_replaces_known_malformed_markers(self, marker_format: str, name: str) -> None:
        """Ошибочные формы известных серверных эмодзи заменяются по точному имени."""
        emoji_ids = {**self.EMOJI_IDS, "ServerLaugh42": "300"}
        text = marker_format.format(name=name)

        assert replace_emojis(text, emoji_ids) == f"<:{name}:{emoji_ids[name]}>"

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    @pytest.mark.parametrize("name", ["unknown", "gachi1", "GACHI1", "Gachi1_extra"])
    def test_unknown_malformed_markers_unchanged(self, marker_format: str, name: str) -> None:
        """Неизвестные имена, другой регистр и более длинные имена не исправляются."""
        text = f"текст {marker_format.format(name=name)} конец"

        assert replace_emojis(text, self.EMOJI_IDS) == text

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    @pytest.mark.parametrize("emoji_ids", [{}, {"Gachi1": ""}], ids=["empty-cache", "empty-id"])
    def test_malformed_markers_without_cached_id_unchanged(
        self, marker_format: str, emoji_ids: dict[str, str]
    ) -> None:
        """Без ID из переданного кэша ошибочный маркер сохраняется целиком."""
        text = f"текст {marker_format.format(name='Gachi1')} конец"

        assert replace_emojis(text, emoji_ids) == text

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    def test_prompt_emoji_without_server_cache_entry_unchanged(self, marker_format: str) -> None:
        """Наличие имени в стандартном промпте не разрешает замену без серверного ID."""
        text = marker_format.format(name="Gachi1")

        assert replace_emojis(text, {"ServerLaugh42": "300"}) == text

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    def test_malformed_markers_use_only_supplied_guild_cache(self, marker_format: str) -> None:
        """ID и разрешённые имена не смешиваются между серверными словарями."""
        known_marker = marker_format.format(name="Gachi1")
        foreign_marker = marker_format.format(name="ForeignEmoji")
        text = f"{known_marker} {foreign_marker}"

        assert replace_emojis(text, {"Gachi1": "100"}) == f"<:Gachi1:100> {foreign_marker}"
        assert replace_emojis(text, {"Gachi1": "200", "ForeignEmoji": "300"}) == (
            "<:Gachi1:200> <:ForeignEmoji:300>"
        )

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    @pytest.mark.parametrize(
        ("prefix", "suffix"),
        [("", ""), ("привет ", " пока"), ("(", "),"), ("«", "»!"), ("\t", "\n")],
        ids=["whole-string", "sentence", "parentheses", "quotes", "whitespace"],
    )
    def test_malformed_markers_preserve_surrounding_text(
        self, marker_format: str, prefix: str, suffix: str
    ) -> None:
        """Замена сохраняет окружающий текст, пунктуацию, табуляцию и переводы строк."""
        marker = marker_format.format(name="Gachi1")

        assert replace_emojis(f"{prefix}{marker}{suffix}", self.EMOJI_IDS) == (
            f"{prefix}<:Gachi1:469464559959277578>{suffix}"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "prefixe:Gachi1",
            "prefix:e:Gachi1",
            "prefix:Gachi1:",
            "e:Gachi1suffix",
            ":Gachi1:suffix",
            "e:Gachi1/path",
            "/e:Gachi1",
            r"C:\e:Gachi1",
            "<e:Gachi1>",
            ":Gachi1:123",
            ":e:Gachi1:",
            "[e:Gachi1",
            "e:Gachi1]",
            "[e: Gachi1]",
            "[e:Gachi1 ]",
            "Gachi1 :Gachi1",
        ],
    )
    def test_partial_and_unsupported_markers_unchanged(self, text: str) -> None:
        """Фрагменты слов, путей и неподдерживаемых форматов не угадываются."""
        assert replace_emojis(text, self.EMOJI_IDS) == text

    @pytest.mark.parametrize("marker_format", MALFORMED_FORMATS)
    @pytest.mark.parametrize("url_prefix", ["http://", "https://", "www.", "HTTPS://"])
    def test_malformed_markers_in_urls_unchanged(
        self, marker_format: str, url_prefix: str
    ) -> None:
        """Маркеры внутри URL сохраняются, а отдельный маркер после ссылки заменяется."""
        marker = marker_format.format(name="Gachi1")
        url = f"{url_prefix}example.com/{marker}?tag={marker}#{marker}"

        assert replace_emojis(f"ссылка {url} потом {marker}", self.EMOJI_IDS) == (
            f"ссылка {url} потом <:Gachi1:469464559959277578>"
        )

    @pytest.mark.parametrize("rendered", ["<:Gachi1:100>", "<a:Gachi1:200>", "<:yoba:300>"])
    def test_rendered_discord_emojis_unchanged(self, rendered: str) -> None:
        """Готовые статичные и анимированные Discord-эмодзи не переписываются."""
        assert replace_emojis(f"{rendered} :e:Gachi1", self.EMOJI_IDS) == (
            f"{rendered} <:Gachi1:469464559959277578>"
        )

    def test_mixed_standard_and_malformed_markers(self) -> None:
        """Правильные неизвестные теги удаляются, ошибочные неизвестные сохраняются."""
        text = "\t[e:yoba], :e:Gachi1! e:yoba; :Gachi1: [e:unknown] :e:unknown e:unknown :unknown:\n"

        assert replace_emojis(text, self.EMOJI_IDS) == (
            "\t<:yoba:1101900451852599427>, <:Gachi1:469464559959277578>! "
            "<:yoba:1101900451852599427>; <:Gachi1:469464559959277578> "
            " :e:unknown e:unknown :unknown:\n"
        )

    def test_does_not_replace_shorter_cached_names(self) -> None:
        """Короткое имя и имя e не подменяют части более длинного ошибочного маркера."""
        emoji_ids = {"Gachi": "100", "Gachi1": "200", "e": "300"}
        text = ":e:Gachi1 e:Gachi1 :Gachi1: :e:unknown"

        assert replace_emojis(text, emoji_ids) == (
            "<:Gachi1:200> <:Gachi1:200> <:Gachi1:200> :e:unknown"
        )

    def test_standard_tags_keep_existing_behavior_in_urls(self) -> None:
        """Защита ошибочных форматов не меняет прежнюю обработку правильных тегов."""
        text = "https://example.com/[e:Gachi1]/[e:unknown]"

        assert replace_emojis(text, self.EMOJI_IDS) == (
            "https://example.com/<:Gachi1:469464559959277578>/"
        )

    def test_repeated_replacement_is_idempotent(self) -> None:
        """Повторная обработка не портит уже заменённые эмодзи и неизвестные маркеры."""
        text = "[e:yoba] :e:Gachi1 :Gachi1: e:Gachi1 :e:unknown <:Gachi1:100>"
        result = replace_emojis(text, self.EMOJI_IDS)

        assert replace_emojis(result, self.EMOJI_IDS) == result


# ── strip_emoji ──────────────────────────────────────────────────


class TestStripEmoji:
    """Тесты для функции strip_emoji."""

    def test_removes_emoticons(self) -> None:
        """Удаляет базовые эмодзи-смайлики."""
        result = strip_emoji("Привет 😂 как дела")
        assert "😂" not in result
        assert "Привет" in result
        assert "как дела" in result

    def test_removes_symbols(self) -> None:
        """Удаляет символьные эмодзи (огонь, звёзды и т.п.)."""
        result = strip_emoji("это 🔥 круто 🎉")
        assert "🔥" not in result
        assert "🎉" not in result
        assert "это" in result
        assert "круто" in result

    def test_removes_flags(self) -> None:
        """Удаляет флаги-эмодзи."""
        result = strip_emoji("Россия 🇷🇺 топ")
        assert "🇷🇺" not in result
        assert "Россия" in result

    def test_plain_text_unchanged(self) -> None:
        """Обычный текст без эмодзи не меняется."""
        result = strip_emoji("просто текст без эмодзи")
        assert result == "просто текст без эмодзи"

    def test_multiple_emoji_in_row(self) -> None:
        """Несколько эмодзи подряд удаляются."""
        result = strip_emoji("😂😂😂 lol")
        assert "😂" not in result
        assert "lol" in result

    def test_empty_string(self) -> None:
        """Пустая строка — пустая строка."""
        assert strip_emoji("") == ""

    def test_only_emoji(self) -> None:
        """Строка только из эмодзи возвращает пустую строку."""
        result = strip_emoji("😂🔥🎉")
        assert result == ""

    def test_returns_string(self) -> None:
        """Всегда возвращает строку."""
        assert isinstance(strip_emoji("тест 🔥"), str)

    def test_removes_custom_discord_emoji(self) -> None:
        """Удаляет кастомные статичные Discord-эмодзи <:name:id>."""
        result = strip_emoji("привет <:pepe:123456789> пока")
        assert "<:pepe:123456789>" not in result
        assert "привет" in result
        assert "пока" in result

    def test_removes_animated_discord_emoji(self) -> None:
        """Удаляет анимированные Discord-эмодзи <a:name:id>."""
        result = strip_emoji("ого <a:dance:987654321> вот это да")
        assert "<a:dance:987654321>" not in result
        assert "ого" in result
        assert "вот это да" in result

    def test_removes_mixed_emoji(self) -> None:
        """Удаляет и Unicode-эмодзи, и кастомные Discord-эмодзи одновременно."""
        result = strip_emoji("текст 😂 и <:kek:111> конец")
        assert "😂" not in result
        assert "<:kek:111>" not in result
        assert "текст" in result
        assert "конец" in result
