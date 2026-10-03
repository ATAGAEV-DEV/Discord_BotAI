"""Unit-тесты для app/tools/prompt.py."""

from app.tools.prompt import EMOJI_PROMPT, ROAST_PROMPT, ROAST_USER_PROMPT, SYSTEM_PROMPT


class TestConstants:
    """Тесты для констант в prompt.py."""

    def test_system_prompt_has_placeholder(self) -> None:
        """SYSTEM_PROMPT содержит плейсхолдер {user_info}."""
        assert "{user_info}" in SYSTEM_PROMPT

    def test_system_prompt_has_only_emoji_section_placeholder(self) -> None:
        """Основной шаблон не содержит безусловных правил или списка эмодзи."""
        assert SYSTEM_PROMPT.count("{emoji_section}") == 1
        assert "{emoji_list}" not in SYSTEM_PROMPT
        assert "эмодзи" not in SYSTEM_PROMPT
        assert "[e:" not in SYSTEM_PROMPT

    def test_emoji_prompt_has_rules_and_list_placeholder(self) -> None:
        """Отдельный шаблон содержит правила формата и место для серверного списка."""
        assert EMOJI_PROMPT.count("{emoji_list}") == 1
        assert EMOJI_PROMPT.count("Использовать только эмодзи сервера из списка ниже.") == 1
        assert EMOJI_PROMPT.count("Правила формата эмодзи:") == 1
        assert EMOJI_PROMPT.count("Доступные эмодзи:") == 1
        assert "[e:ИМЯ]" in EMOJI_PROMPT
        assert "{user_info}" not in EMOJI_PROMPT

    def test_roast_prompt_has_optional_user_section(self) -> None:
        """Базовый roast prompt не обещает пользовательские данные без отдельного блока."""
        assert ROAST_PROMPT.count("{user_section}") == 1
        assert "{user_info}" not in ROAST_PROMPT
        assert "You have access to" not in ROAST_PROMPT
        assert "intimate knowledge" not in ROAST_PROMPT
        assert ROAST_USER_PROMPT.count("{user_info}") == 1
        assert "You have access to" in ROAST_USER_PROMPT
