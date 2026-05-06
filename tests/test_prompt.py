"""Unit-тесты для app/tools/prompt.py."""

from app.tools.prompt import SYSTEM_PROMPT


class TestConstants:
    """Тесты для констант в prompt.py."""

    def test_system_prompt_has_placeholder(self) -> None:
        """SYSTEM_PROMPT содержит плейсхолдер {user_info}."""
        assert "{user_info}" in SYSTEM_PROMPT
