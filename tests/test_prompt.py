"""Unit-тесты для app/tools/prompt.py."""

from app.tools.prompt import (
    SYSTEM_PROMPT,
    system_birthday_prompt,
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


class TestSystemBirthdayPrompt:
    """Тесты для функции system_birthday_prompt."""

    def test_birthday_prompt_not_empty(self) -> None:
        """system_birthday_prompt возвращает непустую строку."""
        descriptions = {"test_user": "Тестовый пользователь"}
        result = system_birthday_prompt(descriptions)
        assert len(result.strip()) > 0

    def test_birthday_prompt_contains_user(self) -> None:
        """system_birthday_prompt включает описание пользователя."""
        descriptions = {"atagaev": "Арби, создатель бота"}
        result = system_birthday_prompt(descriptions)
        assert "atagaev" in result
        assert "Арби" in result

    def test_birthday_prompt_returns_string(self) -> None:
        """Всегда возвращает строку."""
        assert isinstance(system_birthday_prompt({}), str)


class TestConstants:
    """Тесты для констант в prompt.py."""

    def test_system_prompt_has_placeholder(self) -> None:
        """SYSTEM_PROMPT содержит плейсхолдер {user_info}."""
        assert "{user_info}" in SYSTEM_PROMPT
