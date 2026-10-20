"""Unit-тесты cog AIModels."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.cogs.ai_models import AIModels
from app.core.bot import DisBot
from app.data.repositories.guild_ai_config import GuildAIConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bot() -> MagicMock:
    bot = MagicMock(spec=DisBot)
    bot.add_cog = AsyncMock()
    return bot


def _make_ctx(guild_id: int = 123) -> MagicMock:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    ctx.guild = MagicMock()
    ctx.guild.id = guild_id
    return ctx


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_registers_cog() -> None:
    """Setup регистрирует AIModels с ботом."""
    from app.cogs import ai_models as ai_models_module

    bot = _make_bot()
    await ai_models_module.setup(bot)
    bot.add_cog.assert_awaited_once()
    cog = bot.add_cog.await_args.args[0]
    assert isinstance(cog, AIModels)
    assert cog.bot is bot


# ---------------------------------------------------------------------------
# model_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_list_empty() -> None:
    """model_list сообщает что список пуст."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.get_all", new_callable=AsyncMock, return_value=[]):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_list_command.callback(cog, ctx)

    ctx.send.assert_awaited_once()
    assert "пуст" in ctx.send.await_args.args[0]


@pytest.mark.asyncio
async def test_model_list_shows_models() -> None:
    """model_list выводит пронумерованный список моделей."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.get_all", new_callable=AsyncMock,
               return_value=["model-a", "model-b"]):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_list_command.callback(cog, ctx)

    text = ctx.send.await_args.args[0]
    assert "model-a" in text
    assert "model-b" in text


# ---------------------------------------------------------------------------
# model_add
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_add_success() -> None:
    """model_add передаёт имя в репозиторий и выводит ответ."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.add", new_callable=AsyncMock,
               return_value="✅ Модель `gpt-x` добавлена."):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_add_command.callback(cog, ctx, name="gpt-x")

    ctx.send.assert_awaited_once_with("✅ Модель `gpt-x` добавлена.")


# ---------------------------------------------------------------------------
# model_remove
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_remove_success() -> None:
    """model_remove передаёт имя в репозиторий и выводит ответ."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.remove", new_callable=AsyncMock,
               return_value="✅ Модель `gpt-x` удалена."):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_remove_command.callback(cog, ctx, name="gpt-x")

    ctx.send.assert_awaited_once_with("✅ Модель `gpt-x` удалена.")


# ---------------------------------------------------------------------------
# model_set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_set_invalid_type() -> None:
    """model_set отклоняет неверный тип модели."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    await cog.model_set_command.callback(cog, ctx, model_type="bad", name="gpt-x")

    text = ctx.send.await_args.args[0]
    assert "❌" in text
    assert "chat" in text
    assert "mini" in text
    assert "embedding" not in text


@pytest.mark.asyncio
async def test_model_set_model_not_in_list() -> None:
    """model_set отказывает если модели нет в справочнике."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.exists", new_callable=AsyncMock, return_value=False):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_set_command.callback(cog, ctx, model_type="chat", name="unknown-model")

    text = ctx.send.await_args.args[0]
    assert "❌" in text
    assert "unknown-model" in text


@pytest.mark.asyncio
async def test_model_set_chat_success() -> None:
    """model_set chat обновляет chat_model и сообщает об успехе."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx(guild_id=42)

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.exists", new_callable=AsyncMock, return_value=True), \
         patch("app.cogs.ai_models.update_model", new_callable=AsyncMock):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_set_command.callback(cog, ctx, model_type="chat", name="gpt-x")

    text = ctx.send.await_args.args[0]
    assert "✅" in text
    assert "gpt-x" in text


@pytest.mark.asyncio
async def test_model_set_mini_success() -> None:
    """model_set mini обновляет mini_model и сообщает об успехе."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx(guild_id=42)

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.exists", new_callable=AsyncMock, return_value=True), \
         patch("app.cogs.ai_models.update_model", new_callable=AsyncMock):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_set_command.callback(cog, ctx, model_type="mini", name="gpt-mini")

    text = ctx.send.await_args.args[0]
    assert "✅" in text
    assert "gpt-mini" in text


@pytest.mark.asyncio
async def test_model_set_embedding_is_rejected() -> None:
    """model_set не позволяет заменять embedding-модель."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx()

    with patch("app.cogs.ai_models.update_model", new_callable=AsyncMock) as update_model:
        await cog.model_set_command.callback(cog, ctx, model_type="embedding", name="embed-x")

    update_model.assert_not_awaited()
    text = ctx.send.await_args.args[0]
    assert "❌" in text
    assert "chat" in text
    assert "mini" in text


@pytest.mark.asyncio
async def test_model_set_config_error() -> None:
    """model_set выводит ошибку если update_model выбрасывает GuildAIConfigError."""
    bot = _make_bot()
    cog = AIModels(bot)
    ctx = _make_ctx(guild_id=42)

    with patch("app.cogs.ai_models.async_session") as mock_cm, \
         patch("app.cogs.ai_models.ai_models.exists", new_callable=AsyncMock, return_value=True), \
         patch("app.cogs.ai_models.update_model",
               new_callable=AsyncMock,
               side_effect=GuildAIConfigError("AI-конфигурация гильдии 42 не найдена")):
        mock_cm.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_cm.return_value.__aexit__ = AsyncMock(return_value=False)
        await cog.model_set_command.callback(cog, ctx, model_type="chat", name="gpt-x")

    text = ctx.send.await_args.args[0]
    assert "❌" in text

