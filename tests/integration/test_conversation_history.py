"""Проверки хранения и восстановления диалоговых пар в PostgreSQL."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.core.config import CONVERSATION_MAX_PAIRS
from app.data import models
from app.data.repositories import conversation_history
from tests.integration.conftest import Databases


@pytest.mark.integration
async def test_pairs_are_committed_and_loaded_per_channel(databases: Databases) -> None:
    """PostgreSQL сохраняет целые пары и ограничивает выборку отдельно для каждого канала."""
    identities = [(10, 11), (10, 12), (20, 21), (10, 11), (20, 21), (30, 31), (10, 11)]
    for number, (guild_id, channel_id) in enumerate(identities):
        await conversation_history.save_pair(
            guild_id, channel_id, f"[Пользователь: Alice] q{number}", f"a{number}"
        )

    saved = await databases.rows(models.ConversationPair)
    assert len(saved) == len(identities)
    contents = [
        (row.user_content, row.assistant_content) for row in sorted(saved, key=lambda r: r.id)
    ]
    assert contents == [
        (f"[Пользователь: Alice] q{number}", f"a{number}")
        for number in range(len(identities))
    ]
    recent = await conversation_history.load_recent([10, 20], max_pairs=2)
    assert [(row.channel_id, row.assistant_content) for row in recent] == [
        (11, "a3"), (11, "a6"), (12, "a1"), (21, "a2"), (21, "a4")
    ]


@pytest.mark.integration
async def test_cache_restore_preserves_recent_pair_chronology(
    databases: Databases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ребут восстанавливает лимит пар по ID, даже когда время записей совпадает."""
    from app.services import llama_integration

    total = CONVERSATION_MAX_PAIRS + 3
    await databases.seed(
        *[
            models.ConversationPair(
                guild_id=10,
                channel_id=11,
                user_content=f"q{number}",
                assistant_content=f"a{number}",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for number in range(total)
        ],
        models.ConversationPair(
            guild_id=10, channel_id=12, user_content="другой канал", assistant_content="ответ"
        ),
        models.ConversationPair(
            guild_id=20, channel_id=21, user_content="другой сервер", assistant_content="ответ"
        ),
    )
    monkeypatch.setattr(llama_integration.chromadb, "PersistentClient", MagicMock())
    manager = llama_integration.LlamaIndexManager()

    await manager.load_channel_histories([10, 20])

    history = manager.get_channel_history(11)
    expected = [
        message
        for number in range(3, total)
        for message in (
            {"role": "user", "content": f"q{number}"},
            {"role": "assistant", "content": f"a{number}"},
        )
    ]
    assert history.get_messages() == expected
    assert manager.get_channel_history(12).get_texts() == ["другой канал", "ответ"]
    assert manager.get_channel_history(21).get_texts() == ["другой сервер", "ответ"]
    assert manager._channel_server_map == {11: 10, 12: 10, 21: 20}

    history.add_pair({"role": "user", "content": "новый"}, {"role": "assistant", "content": "ответ"})
    before_reconnect = history.get_messages()
    await manager.load_channel_histories([10, 20])
    assert manager.get_channel_history(11) is history
    assert history.get_messages() == before_reconnect
    assert len(await databases.rows(models.ConversationPair)) == total + 2
