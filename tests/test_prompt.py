"""Unit-тесты для app/tools/prompt.py."""

from app.tools.prompt import (
    SYSTEM_BIRTHDAY_PROMPT,
    SYSTEM_PROMPT,
    USER_DESCRIPTIONS,
    system_holiday_prompt,
)


class TestSystemHolidayPrompt:
    """Тесты для функции system_holiday_prompt."""

    def test_contains_holiday_name(self) -> None:
        """Промпт содержит название праздника."""
        result = system_holiday_prompt("Новым годом")
        assert "Новым годом" in result

    def test_contains_emoji_list(self) -> None:
        """Промпт содержит список эмодзи."""
        result = system_holiday_prompt("Днем Бичей")
        assert ":yoba:" in result
        assert ":Gachi1:" in result

    def test_returns_string(self) -> None:
        """Всегда возвращает строку."""
        assert isinstance(system_holiday_prompt("тест"), str)


class TestConstants:
    """Тесты для констант в prompt.py."""

    def test_user_descriptions_not_empty(self) -> None:
        """USER_DESCRIPTIONS не пустой."""
        assert len(USER_DESCRIPTIONS) > 0

    def test_system_prompt_has_placeholder(self) -> None:
        """SYSTEM_PROMPT содержит плейсхолдер {user_info}."""
        assert "{user_info}" in SYSTEM_PROMPT

    def test_birthday_prompt_not_empty(self) -> None:
        """SYSTEM_BIRTHDAY_PROMPT не пустой."""
        assert len(SYSTEM_BIRTHDAY_PROMPT.strip()) > 0
