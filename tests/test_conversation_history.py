"""Проверки постоянного хранения диалоговых пар без внешней БД."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.config import CONVERSATION_MAX_PAIRS
from app.data import decorators, models


@pytest.fixture
def history_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Исполняет SQL репозитория в SQLite через интерфейс async-сессии."""
    engine = create_engine("sqlite+pysqlite://", creator=lambda: sqlite3.connect(":memory:"))
    models.Base.metadata.create_all(engine)

    @asynccontextmanager
    async def session_context() -> AsyncIterator[MagicMock]:
        with Session(engine) as session:
            adapter = MagicMock(spec=AsyncSession)
            adapter.add.side_effect = session.add
            adapter.execute = AsyncMock(side_effect=session.execute)
            adapter.commit = AsyncMock(side_effect=session.commit)
            yield adapter

    monkeypatch.setattr(decorators, "async_session", session_context)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_save_pair_commits_both_contents(history_database: Engine) -> None:
    """Одна завершённая пара сохраняется целиком без изменения текстов."""
    from app.data.repositories import conversation_history

    await conversation_history.save_pair(
        123, 456, "[Пользователь: Alice] Как дела, @Bob?", "Ответ бота"
    )

    with Session(history_database) as session:
        rows = session.scalars(select(models.ConversationPair)).all()
        assert len(rows) == 1
        assert rows[0].guild_id == 123
        assert rows[0].channel_id == 456
        assert rows[0].user_content == "[Пользователь: Alice] Как дела, @Bob?"
        assert rows[0].assistant_content == "Ответ бота"
        assert rows[0].created_at is not None


@pytest.mark.asyncio
async def test_load_recent_limits_each_channel_and_preserves_pair_order(
    history_database: Engine,
) -> None:
    """Последние пары каждого канала возвращаются от старых к новым даже при равном времени."""
    from app.data.repositories import conversation_history

    identities = [
        (1, 10, 11),
        (2, 10, 12),
        (3, 20, 21),
        (4, 10, 11),
        (5, 20, 21),
        (6, 30, 31),
        (7, 10, 11),
        (8, 10, 12),
    ]
    with Session(history_database) as session:
        session.add_all(
            models.ConversationPair(
                id=pair_id,
                guild_id=guild_id,
                channel_id=channel_id,
                user_content=f"q{pair_id}",
                assistant_content=f"a{pair_id}",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for pair_id, guild_id, channel_id in reversed(identities)
        )
        session.commit()

    rows = await conversation_history.load_recent([10, 20], max_pairs=2)

    actual = [
        (row.guild_id, row.channel_id, row.user_content, row.assistant_content) for row in rows
    ]
    assert actual == [
        (10, 11, "q4", "a4"),
        (10, 11, "q7", "a7"),
        (10, 12, "q2", "a2"),
        (10, 12, "q8", "a8"),
        (20, 21, "q3", "a3"),
        (20, 21, "q5", "a5"),
    ]


@pytest.mark.asyncio
async def test_load_channel_histories_restores_separate_channel_buffers(
    history_database: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогрев кэша восстанавливает роли, хронологию и связь каналов с серверами."""
    from app.data.repositories import conversation_history
    from app.services import llama_integration

    for guild_id, channel_id, question, answer in [
        (10, 11, "первый вопрос", "первый ответ"),
        (20, 21, "другой сервер", "ответ другого сервера"),
        (10, 12, "другой канал", "ответ другого канала"),
        (10, 11, "второй вопрос", "второй ответ"),
    ]:
        await conversation_history.save_pair(guild_id, channel_id, question, answer)

    monkeypatch.setattr(llama_integration.chromadb, "PersistentClient", MagicMock())
    manager = llama_integration.LlamaIndexManager()
    save_pair = AsyncMock()
    monkeypatch.setattr(conversation_history, "save_pair", save_pair)
    index_messages = AsyncMock()
    monkeypatch.setattr(manager, "index_messages", index_messages)

    await manager.load_channel_histories([10, 20])

    assert manager.get_channel_history(11).get_messages() == [
        {"role": "user", "content": "первый вопрос"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "второй вопрос"},
        {"role": "assistant", "content": "второй ответ"},
    ]
    assert manager.get_channel_history(12).get_texts() == ["другой канал", "ответ другого канала"]
    assert manager.get_channel_history(21).get_texts() == [
        "другой сервер", "ответ другого сервера"
    ]
    assert manager._channel_server_map == {11: 10, 12: 10, 21: 20}
    save_pair.assert_not_awaited()
    index_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_channel_history_waits_for_pending_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Первый запрос ждёт восстановления, а не получает пустую историю во время прогрева."""
    from app.data.repositories import conversation_history
    from app.services import llama_integration

    started = asyncio.Event()
    release = asyncio.Event()

    async def load_recent(guild_ids: list[int]) -> list[models.ConversationPair]:
        started.set()
        await release.wait()
        return [
            models.ConversationPair(
                id=1, guild_id=10, channel_id=11, user_content="вопрос", assistant_content="ответ"
            )
        ]

    load = AsyncMock(side_effect=load_recent)
    monkeypatch.setattr(conversation_history, "load_recent", load)
    monkeypatch.setattr(llama_integration.chromadb, "PersistentClient", MagicMock())
    manager = llama_integration.LlamaIndexManager()
    load_task = asyncio.create_task(manager.load_channel_histories([10]))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        request_task = asyncio.create_task(manager.ensure_channel_history(11, 10))
        await asyncio.sleep(0)
        assert not request_task.done()
    finally:
        release.set()
        await load_task

    history = await request_task
    assert history.get_texts() == ["вопрос", "ответ"]
    load.assert_awaited_once_with([10])


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Подменяет только AI и RAG, оставляя настоящие историю и SQL-репозиторий."""
    from app.services import llama_integration

    monkeypatch.setattr(llama_integration.chromadb, "PersistentClient", MagicMock())
    from app.core import handlers

    def new_manager() -> llama_integration.LlamaIndexManager:
        """Создаёт пустой кэш, не открывая ChromaDB."""
        manager = llama_integration.LlamaIndexManager()
        manager.query_relevant_context = AsyncMock(return_value=["архивный фрагмент"])
        manager.index_messages = AsyncMock()
        return manager

    live_manager = new_manager()
    monkeypatch.setattr(handlers, "llama_manager", live_manager)
    monkeypatch.setattr(handlers, "user_prompt", lambda *args, **kwargs: "правила")
    monkeypatch.setattr(handlers.user_descriptions_cache, "get", lambda guild_id: {})
    monkeypatch.setattr(handlers, "get_model", lambda guild_id: "test-model")
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ"))]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(handlers, "get_client", lambda guild_id: client)
    return SimpleNamespace(
        handlers=handlers, manager=live_manager, create=create, new_manager=new_manager
    )


@pytest.mark.asyncio
async def test_ai_generate_round_trip_keeps_identical_model_messages(
    history_database: Engine, generation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сохранённая пара после ребута даёт модели тот же запрос, что и живая история."""
    handlers = generation.handlers
    live_manager = generation.manager
    create = generation.create
    create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        for text in ["Первый ответ", "Продолжение", "Продолжение"]
    ]

    assert await handlers.ai_generate(
        "Как дела, <@5>?", 10, "Alice", mentions={5: "Bob"}, channel_id=11
    ) == "Первый ответ"

    with Session(history_database) as session:
        pair = session.scalar(select(models.ConversationPair))
        assert pair is not None
        assert pair.user_content == "[Пользователь: Alice] Как дела, @Bob?"
        assert pair.assistant_content == "Первый ответ"

    restored_manager = generation.new_manager()
    await restored_manager.load_channel_histories([10])
    assert restored_manager.get_channel_history(11).get_messages() == (
        live_manager.get_channel_history(11).get_messages()
    )

    await handlers.ai_generate("Расскажи подробнее", 10, "Alice", channel_id=11)
    live_messages = create.await_args.kwargs["messages"]
    monkeypatch.setattr(handlers, "llama_manager", restored_manager)
    await handlers.ai_generate("Расскажи подробнее", 10, "Alice", channel_id=11)
    restored_messages = create.await_args.kwargs["messages"]
    await asyncio.sleep(0)

    assert restored_messages == live_messages
    assert [message["role"] for message in restored_messages] == [
        "system", "user", "assistant", "user"
    ]
    assert restored_messages[1:3] == [
        {"role": "user", "content": "[Пользователь: Alice] Как дела, @Bob?"},
        {"role": "assistant", "content": "Первый ответ"},
    ]
    assert "### Архивные фрагменты" in restored_messages[-1]["content"]
    assert "### Текущий вопрос" in restored_messages[-1]["content"]


@pytest.mark.asyncio
async def test_restore_uses_default_limit_without_replacing_live_cache_on_reconnect(
    history_database: Engine, generation: SimpleNamespace
) -> None:
    """Прогрев берёт последние CONVERSATION_MAX_PAIRS и не стирает новые реплики при reconnect."""
    total = CONVERSATION_MAX_PAIRS + 3
    with Session(history_database) as session:
        session.add_all(
            models.ConversationPair(
                guild_id=10, channel_id=11, user_content=f"q{number}", assistant_content=f"a{number}"
            )
            for number in range(total)
        )
        session.commit()

    manager = generation.manager
    await manager.load_channel_histories([10])
    history = manager.get_channel_history(11)
    assert history.get_texts() == [
        text for number in range(3, total) for text in (f"q{number}", f"a{number}")
    ]
    history.add_pair({"role": "user", "content": "новый"}, {"role": "assistant", "content": "ответ"})
    before_reconnect = history.get_messages()

    await manager.load_channel_histories([10])

    assert manager.get_channel_history(11) is history
    assert history.get_messages() == before_reconnect


@pytest.mark.asyncio
async def test_failed_restore_does_not_publish_partial_cache(
    generation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сбой загрузки сохраняет прежний кэш и не блокирует последующую попытку прогрева."""
    from app.data.repositories import conversation_history

    manager = generation.manager
    old_history = manager.get_channel_history(11, 10)
    old_history.add_pair(
        {"role": "user", "content": "старый"}, {"role": "assistant", "content": "ответ"}
    )
    rows = [
        models.ConversationPair(
            id=1, guild_id=10, channel_id=11, user_content="из БД", assistant_content="ответ из БД"
        )
    ]
    load = AsyncMock(side_effect=[RuntimeError("database unavailable"), rows])
    monkeypatch.setattr(conversation_history, "load_recent", load)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await manager.load_channel_histories([10])

    assert manager.get_channel_history(11) is old_history
    assert old_history.get_texts() == ["старый", "ответ"]
    assert manager._loaded_history_servers == set()
    await manager.load_channel_histories([10])
    assert manager.get_channel_history(11).get_texts() == ["из БД", "ответ из БД"]
    assert load.await_count == 2


@pytest.mark.asyncio
async def test_failed_save_keeps_completed_answer_in_cache(
    history_database: Engine,
    generation: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ошибка БД не заменяет готовый ответ ошибкой и не теряет живую краткосрочную память."""
    from app.data.repositories import conversation_history

    monkeypatch.setattr(conversation_history, "save_pair", AsyncMock(side_effect=RuntimeError))

    response = await generation.handlers.ai_generate("вопрос", 10, "Alice", channel_id=11)
    await asyncio.sleep(0)

    assert response == "Ответ"
    assert generation.manager.get_channel_history(11).get_messages() == [
        {"role": "user", "content": "[Пользователь: Alice] вопрос"},
        {"role": "assistant", "content": "Ответ"},
    ]
    with Session(history_database) as session:
        assert session.scalars(select(models.ConversationPair)).all() == []
    assert "Ошибка сохранения пары диалога канала 11: RuntimeError" in capsys.readouterr().out
    generation.manager.index_messages.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
async def test_generation_failure_does_not_save_incomplete_pair(
    history_database: Engine, generation: SimpleNamespace, error_type: type[Exception]
) -> None:
    """Ошибка или таймаут API не создают вопрос без ответа в БД или в кэше."""
    generation.create.side_effect = error_type

    if error_type is TimeoutError:
        result = await generation.handlers.ai_generate("вопрос", 10, "Alice", channel_id=11)
        assert "⏳" in result
    else:
        with pytest.raises(error_type):
            await generation.handlers.ai_generate("вопрос", 10, "Alice", channel_id=11)

    assert generation.manager.get_channel_history(11).get_messages() == []
    with Session(history_database) as session:
        assert session.scalars(select(models.ConversationPair)).all() == []
    generation.manager.index_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_database_save_does_not_consume_generation_timeout(
    history_database: Engine, generation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Задержка записи после успешной генерации не превращает готовый ответ в таймаут AI."""
    from app.data.repositories import conversation_history

    save = conversation_history.save_pair

    async def slow_save(*args: object) -> None:
        """Задерживает только сохранение, не работу модели."""
        await asyncio.sleep(0.15)
        await save(*args)

    monkeypatch.setattr(conversation_history, "save_pair", slow_save)
    monkeypatch.setattr(generation.handlers, "AI_GENERATE_TIMEOUT", 0.1)

    result = await generation.handlers.ai_generate("вопрос", 10, "Alice", channel_id=11)
    await asyncio.sleep(0)

    assert result == "Ответ"
    with Session(history_database) as session:
        assert len(session.scalars(select(models.ConversationPair)).all()) == 1


@pytest.mark.asyncio
async def test_concurrent_saves_keep_database_and_cache_in_the_same_order(
    history_database: Engine, generation: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Второй ответ того же канала не обгоняет незавершённое сохранение первого."""
    from app.data.repositories import conversation_history

    save = conversation_history.save_pair
    first_save_started = asyncio.Event()
    release_first_save = asyncio.Event()
    second_generated = asyncio.Event()

    async def blocked_save(guild_id: int, channel_id: int, question: str, answer: str) -> None:
        """Позволяет второму ответу завершиться до фиксации первого."""
        if question.endswith("первый"):
            first_save_started.set()
            await release_first_save.wait()
        await save(guild_id, channel_id, question, answer)

    async def complete(**kwargs: object) -> object:
        """Отмечает момент готовности второго ответа."""
        if generation.create.await_count == 2:
            second_generated.set()
        return generation.create.return_value

    save_mock = AsyncMock(side_effect=blocked_save)
    monkeypatch.setattr(conversation_history, "save_pair", save_mock)
    generation.create.side_effect = complete
    first = asyncio.create_task(
        generation.handlers.ai_generate("первый", 10, "Alice", channel_id=11)
    )
    tasks = [first]
    try:
        await asyncio.wait_for(first_save_started.wait(), timeout=5)
        tasks.append(asyncio.create_task(
            generation.handlers.ai_generate("второй", 10, "Bob", channel_id=11)
        ))
        await asyncio.wait_for(second_generated.wait(), timeout=5)
        await asyncio.sleep(0)
        assert save_mock.await_count == 1
    finally:
        release_first_save.set()
        await asyncio.gather(*tasks)
    await asyncio.sleep(0)

    with Session(history_database) as session:
        rows = session.scalars(select(models.ConversationPair).order_by(models.ConversationPair.id))
        stored = [text for row in rows for text in (row.user_content, row.assistant_content)]
    assert stored == generation.manager.get_channel_history(11).get_texts()
    assert stored == ["[Пользователь: Alice] первый", "Ответ", "[Пользователь: Bob] второй", "Ответ"]
