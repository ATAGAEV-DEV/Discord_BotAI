"""Unit-тесты repository guild_ai_configs и fail-closed bootstrap."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import AISettings
from app.data import guild_ai_config_cache
from app.data.models import GuildAIConfig
from app.data.repositories import guild_ai_config


def _settings(**overrides: object) -> AISettings:
    """Возвращает полную тестовую конфигурацию."""
    values: dict[str, object] = {
        "provider": "custom-chat-provider",
        "token": "chat-secret",
        "embedding_token": "embedding-secret",
        "api_url": "https://chat.example/v1",
        "embedding_provider": "custom-embedding-provider",
        "embedding_api_url": "https://embedding.example/v1",
        "chat_model": "chat-model",
        "mini_model": "mini-model",
        "embedding_model": "embedding-model",
    }
    values.update(overrides)
    return AISettings(**values)


def _row(guild_id: int = 1, **overrides: object) -> GuildAIConfig:
    """Создаёт ORM row с обеими credentials."""
    values = {
        "guild_id": guild_id,
        "provider": "custom-chat-provider",
        "api_url": "https://chat.example/v1",
        "token": "stored-chat-secret",
        "chat_model": "stored-chat-model",
        "mini_model": "stored-mini-model",
        "embedding_provider": "custom-embedding-provider",
        "embedding_api_url": "https://stored-embedding.example/v1",
        "embedding_token": "stored-embedding-secret",
        "embedding_model": "stored-embedding-model",
    }
    values.update(overrides)
    return GuildAIConfig(**values)


class _Result:
    """Минимальный SQLAlchemy result для repository tests."""

    def __init__(self, row: GuildAIConfig | None) -> None:
        self.row = row

    def scalar_one_or_none(self) -> GuildAIConfig | None:
        """Возвращает одну fake строку."""
        return self.row

    def scalars(self) -> SimpleNamespace:
        """Возвращает fake scalar result."""
        return SimpleNamespace(all=lambda: [self.row] if self.row is not None else [])


class _Session:
    """Fake AsyncSession с одной видимой строкой."""

    def __init__(self, row: GuildAIConfig | None = None) -> None:
        self._row = row
        self.pending: GuildAIConfig | None = None
        self.commit = AsyncMock(side_effect=self._do_commit)
        self.rollback = AsyncMock()

    async def execute(self, statement: object) -> _Result:
        """Возвращает текущую строку."""
        del statement
        return _Result(self._row)

    def add(self, instance: GuildAIConfig) -> None:
        """Запоминает pending insert без публикации до commit."""
        self.pending = instance

    async def _do_commit(self) -> None:
        """Публикует строку только после успешного commit."""
        if self.pending is not None:
            self._row = self.pending
            self.pending = None



@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Изолирует config cache и repository locks."""
    guild_ai_config_cache.clear()
    guild_ai_config.clear_locks()
    yield
    guild_ai_config_cache.clear()
    guild_ai_config.clear_locks()


@pytest.mark.asyncio
async def test_existing_complete_row_has_priority_over_environment() -> None:
    """Полная строка БД не заменяется environment settings."""
    session = _Session(_row())

    result = await guild_ai_config.ensure(session, 1, _settings(token="env-chat"))

    assert result.token == "stored-chat-secret"
    assert result.embedding_token == "stored-embedding-secret"
    assert result.api_url == "https://chat.example/v1"
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_requires_embedding_token() -> None:
    """Нельзя создать строку без embedding token."""
    session = _Session()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="embedding token"):
        await guild_ai_config.ensure(session, 2, _settings(embedding_token=None))

    assert session.pending is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_requires_all_models() -> None:
    """Неполная модельная конфигурация не сохраняется в БД."""
    session = _Session()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="chat_model"):
        await guild_ai_config.ensure(session, 6, _settings(chat_model=None))

    assert session.pending is None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_row_is_written_after_commit() -> None:
    """Bootstrap сохраняет credentials после commit."""
    session = _Session()
    settings = _settings()

    result = await guild_ai_config.ensure(session, 3, settings)

    assert result.token == "chat-secret"
    assert result.embedding_token == "embedding-secret"
    assert session._row is not None
    assert session._row.embedding_token == "embedding-secret"
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_commit_failure_does_not_publish_cache_or_partial_settings() -> None:
    """Ошибка commit не оставляет новую конфигурацию активной."""
    session = _Session()
    session.commit = AsyncMock(side_effect=RuntimeError("commit failed"))

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Не удалось сохранить"):
        await guild_ai_config_cache.ensure(session, 4, _settings())

    assert guild_ai_config_cache.get(4) is None
    assert session._row is None
    session.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_bootstrap_uses_one_guild_lock() -> None:
    """Конкурентные вызовы одной гильдии не создают две строки."""
    session = _Session()
    original_commit = session.commit
    calls = 0

    async def delayed_commit() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        await original_commit()

    session.commit = AsyncMock(side_effect=delayed_commit)
    results = await asyncio.gather(
        guild_ai_config_cache.ensure(session, 5, _settings()),
        guild_ai_config_cache.ensure(session, 5, _settings(token="other-chat")),
    )

    assert calls == 1
    assert results[0].token == results[1].token


# ---------------------------------------------------------------------------
# _coerce_environment_settings с Mapping (строки 77, 80-89)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_with_mapping_environment_settings() -> None:
    """ensure принимает dict-подобный mapping вместо AISettings."""
    env_mapping = {
        "provider": "custom-chat-provider",
        "api_url": "https://chat.example/v1",
        "token": "chat-secret",
        "chat_model": "chat-model",
        "mini_model": "mini-model",
        "embedding_provider": "custom-embedding-provider",
        "embedding_api_url": "https://embedding.example/v1",
        "embedding_token": "embedding-secret",
        "embedding_model": "embedding-model",
    }
    session = _Session()

    result = await guild_ai_config.ensure(session, 20, env_mapping)

    assert result.token == "chat-secret"
    assert result.embedding_token == "embedding-secret"
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# IntegrityError в _ensure_unlocked — winner path (строки 137-141)
# ---------------------------------------------------------------------------

class _IntegritySession(_Session):
    """Fake session: IntegrityError на commit, затем победитель из get()."""

    def __init__(self, winner_row: "GuildAIConfig") -> None:
        super().__init__(None)  # нет существующей строки
        self._winner = winner_row
        self._commit_called = 0

        async def _commit_with_integrity() -> None:
            self._commit_called += 1
            from sqlalchemy.exc import IntegrityError
            raise IntegrityError(None, None, Exception())

        self.commit = AsyncMock(side_effect=_commit_with_integrity)

    async def execute(self, statement: object) -> "_Result":
        """После IntegrityError get() должен вернуть победителя."""
        if self._commit_called > 0:
            return _Result(self._winner)
        return _Result(None)


@pytest.mark.asyncio
async def test_ensure_returns_winner_on_integrity_error() -> None:
    """_ensure_unlocked возвращает победителя при IntegrityError (race condition)."""
    winner = _row(guild_id=30)
    session = _IntegritySession(winner)

    result = await guild_ai_config.ensure(session, 30, _settings())

    assert result.token == "stored-chat-secret"


# ---------------------------------------------------------------------------
# IntegrityError когда winner == None (строка 152) → GuildAIConfigError
# ---------------------------------------------------------------------------

class _IntegrityNoWinnerSession(_Session):
    """Fake session: IntegrityError на commit и get() возвращает None."""

    def __init__(self) -> None:
        super().__init__(None)

        async def _commit_with_integrity() -> None:
            from sqlalchemy.exc import IntegrityError
            raise IntegrityError(None, None, Exception())

        self.commit = AsyncMock(side_effect=_commit_with_integrity)


@pytest.mark.asyncio
async def test_ensure_raises_when_integrity_error_and_no_winner() -> None:
    """_ensure_unlocked выбрасывает GuildAIConfigError если нет победителя после IntegrityError."""
    session = _IntegrityNoWinnerSession()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Не удалось сохранить"):
        await guild_ai_config.ensure(session, 31, _settings())


# ---------------------------------------------------------------------------
# _coerce_environment_settings(None) → get_ai_settings() (строка 77)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_uses_env_settings_when_none_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure вызывает get_ai_settings() когда environment_settings=None."""
    from app.core.config import AISettings

    fake_env = _settings()
    monkeypatch.setattr(
        "app.data.repositories.guild_ai_config.get_ai_settings",
        lambda: fake_env,
    )
    session = _Session()

    result = await guild_ai_config.ensure(session, 32, None)

    assert result.token == fake_env.token
    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# _ensure_unlocked — get() None после успешного commit (строка 152)
# ---------------------------------------------------------------------------

class _CommitOkButGetNoneSession(_Session):
    """Commit успешен, но get() всегда возвращает None (не публикует pending)."""

    def __init__(self) -> None:
        super().__init__(None)

        async def _commit_no_publish() -> None:
            # commit не публикует pending — _row остаётся None
            self.pending = None

        self.commit = AsyncMock(side_effect=_commit_no_publish)


@pytest.mark.asyncio
async def test_ensure_raises_when_get_returns_none_after_commit() -> None:
    """_ensure_unlocked выбрасывает GuildAIConfigError когда get() None после commit."""
    session = _CommitOkButGetNoneSession()

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Не удалось сохранить"):
        await guild_ai_config.ensure(session, 33, _settings())


# ---------------------------------------------------------------------------
# update_provider — validate_ai_settings ошибка (строки 246-247)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_provider_raises_when_candidate_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update_provider выбрасывает GuildAIConfigError если новый кандидат не проходит валидацию."""
    session = _UpdateSession(_row(guild_id=9))

    # Патчим validate_ai_settings: для строки из БД пропускаем (первый вызов),
    # для кандидата бросаем (второй вызов)
    call_count = 0
    original_validate = guild_ai_config.validate_ai_settings

    def _validate_side_effect(s: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_validate(s, **kwargs)
        raise ValueError("невалидный URL")

    monkeypatch.setattr(
        "app.data.repositories.guild_ai_config.validate_ai_settings",
        _validate_side_effect,
    )

    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Невозможно обновить"):
        await guild_ai_config.update_provider(
            session, 9, "chat", "openai", "https://api.openai.com/v1", "tok"
        )


# ---------------------------------------------------------------------------
# _row_to_settings (строки 43-46)
# ---------------------------------------------------------------------------

def test_row_to_settings_raises_on_invalid_row() -> None:
    """_row_to_settings выбрасывает GuildAIConfigError для невалидной ORM-строки."""
    row = _row(provider="")  # пустой provider — невалиден
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="конфигурация гильдии"):
        guild_ai_config._row_to_settings(row)


def test_row_to_settings_converts_valid_row() -> None:
    """_row_to_settings преобразует валидную ORM-строку в AISettings."""
    row = _row(guild_id=99)
    settings = guild_ai_config._row_to_settings(row)
    assert settings.token == "stored-chat-secret"
    assert settings.embedding_token == "stored-embedding-secret"
    assert settings.chat_model == "stored-chat-model"


# ---------------------------------------------------------------------------
# get (строки 49-57)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_returns_none_for_missing_guild() -> None:
    """get возвращает None если гильдия не найдена."""
    session = _Session(None)
    result = await guild_ai_config.get(session, 777)
    assert result is None


@pytest.mark.asyncio
async def test_get_returns_settings_for_existing_guild() -> None:
    """get возвращает AISettings для существующей гильдии."""
    session = _Session(_row(guild_id=10))
    result = await guild_ai_config.get(session, 10)
    assert result is not None
    assert result.token == "stored-chat-secret"


# ---------------------------------------------------------------------------
# load_all (строки 60-69)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_all_returns_mapping_for_valid_rows() -> None:
    """load_all возвращает dict guild_id → AISettings для валидных строк."""
    session = _Session(_row(guild_id=1))
    result = await guild_ai_config.load_all(session)
    assert 1 in result
    assert result[1].token == "stored-chat-secret"


@pytest.mark.asyncio
async def test_load_all_skips_invalid_rows() -> None:
    """load_all пропускает строки с ошибочной конфигурацией."""

    class _MultiSession:
        """Fake session с несколькими строками: одна валидная, одна — нет."""

        async def execute(self, _stmt: object) -> object:
            from types import SimpleNamespace
            valid = _row(guild_id=1)
            invalid = _row(guild_id=2, provider="")
            return SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [valid, invalid])
            )

    result = await guild_ai_config.load_all(_MultiSession())
    assert 1 in result
    assert 2 not in result


# ---------------------------------------------------------------------------
# update_model (строки 186-204)
# ---------------------------------------------------------------------------

class _UpdateSession(_Session):
    """Fake session, поддерживающий UPDATE через rowcount."""

    def __init__(self, row: "GuildAIConfig | None" = None, *, found: bool = True) -> None:
        super().__init__(row)
        self._found = found

    async def execute(self, statement: object) -> object:
        stmt_str = str(statement).strip().upper()
        if stmt_str.startswith("UPDATE"):
            from types import SimpleNamespace
            return SimpleNamespace(rowcount=1 if self._found else 0)
        # SELECT — используем базовый _Result с scalar_one_or_none
        return _Result(self._row)


@pytest.mark.asyncio
async def test_update_model_raises_for_invalid_field() -> None:
    """update_model выбрасывает GuildAIConfigError для неверного имени поля."""
    session = _UpdateSession(_row(guild_id=1))
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Некорректное поле"):
        await guild_ai_config.update_model(session, 1, "invalid_field", "gpt-4")


@pytest.mark.asyncio
async def test_update_model_raises_for_empty_model_name() -> None:
    """update_model выбрасывает GuildAIConfigError для пустой строки модели."""
    session = _UpdateSession(_row(guild_id=1))
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="не задана"):
        await guild_ai_config.update_model(session, 1, "chat_model", "  ")


@pytest.mark.asyncio
async def test_update_model_raises_when_guild_not_found() -> None:
    """update_model выбрасывает GuildAIConfigError если строка не найдена."""
    session = _UpdateSession(_row(guild_id=1), found=False)
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="не найдена"):
        await guild_ai_config.update_model(session, 999, "chat_model", "gpt-4")


@pytest.mark.asyncio
async def test_update_model_commits_and_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update_model коммитит и инвалидирует кэш гильдии."""
    session = _UpdateSession(_row(guild_id=5))
    guild_ai_config_cache._cache[5] = _settings()

    # После invalidate/ensure кэш должен переинициализироваться
    # Патчим ensure чтобы он просто возвращал settings, не делая реального SELECT
    async def _fake_ensure(sess: object, gid: int, env: object = None) -> AISettings:
        guild_ai_config_cache._cache[gid] = _settings()
        return _settings()

    monkeypatch.setattr(guild_ai_config_cache, "ensure", _fake_ensure)

    await guild_ai_config.update_model(session, 5, "chat_model", "gpt-4-new")

    session.commit.assert_awaited_once()
    assert guild_ai_config_cache.get(5) is not None


@pytest.mark.asyncio
async def test_update_mini_model_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update_model для mini_model также коммитит."""
    session = _UpdateSession(_row(guild_id=6))
    guild_ai_config_cache._cache[6] = _settings()

    async def _fake_ensure(sess: object, gid: int, env: object = None) -> AISettings:
        guild_ai_config_cache._cache[gid] = _settings()
        return _settings()

    monkeypatch.setattr(guild_ai_config_cache, "ensure", _fake_ensure)

    await guild_ai_config.update_model(session, 6, "mini_model", "gpt-4-mini-new")

    session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# update_provider (строки 207-273)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_provider_raises_for_invalid_type() -> None:
    """update_provider выбрасывает GuildAIConfigError для неверного provider_type."""
    session = _UpdateSession(_row(guild_id=1))
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="Некорректный тип"):
        await guild_ai_config.update_provider(
            session, 1, "unknown-type", "p", "https://p.example/v1", "tok"
        )


@pytest.mark.asyncio
async def test_update_provider_raises_for_empty_name() -> None:
    """update_provider выбрасывает GuildAIConfigError для пустого имени."""
    session = _UpdateSession(_row(guild_id=1))
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="не задано"):
        await guild_ai_config.update_provider(
            session, 1, "chat", "  ", "https://p.example/v1", "tok"
        )


@pytest.mark.asyncio
async def test_update_provider_raises_when_guild_not_found() -> None:
    """update_provider выбрасывает GuildAIConfigError если гильдия не найдена."""
    session = _Session(None)
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="не найдена"):
        await guild_ai_config.update_provider(
            session, 999, "chat", "openai", "https://api.openai.com/v1", "sk-tok"
        )


@pytest.mark.asyncio
async def test_update_provider_raises_when_update_rowcount_is_zero() -> None:
    """update_provider выбрасывает GuildAIConfigError если UPDATE затронул 0 строк."""
    session = _UpdateSession(_row(guild_id=1), found=False)
    session._row = _row(guild_id=1)
    with pytest.raises(guild_ai_config.GuildAIConfigError, match="не найдена"):
        await guild_ai_config.update_provider(
            session, 1, "chat", "openai", "https://api.openai.com/v1", "sk-tok"
        )


@pytest.mark.asyncio
async def test_update_provider_chat_commits_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update_provider для chat-типа коммитит и инвалидирует кэш."""
    session = _UpdateSession(_row(guild_id=7))
    guild_ai_config_cache._cache[7] = _settings()

    async def _fake_ensure(sess: object, gid: int, env: object = None) -> AISettings:
        guild_ai_config_cache._cache[gid] = _settings()
        return _settings()

    monkeypatch.setattr(guild_ai_config_cache, "ensure", _fake_ensure)

    await guild_ai_config.update_provider(
        session, 7, "chat", "openai", "https://api.openai.com/v1", "sk-tok"
    )

    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_provider_embedding_commits_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """update_provider для embedding-типа коммитит и инвалидирует кэш."""
    session = _UpdateSession(_row(guild_id=8))
    guild_ai_config_cache._cache[8] = _settings()

    async def _fake_ensure(sess: object, gid: int, env: object = None) -> AISettings:
        guild_ai_config_cache._cache[gid] = _settings()
        return _settings()

    monkeypatch.setattr(guild_ai_config_cache, "ensure", _fake_ensure)

    await guild_ai_config.update_provider(
        session, 8, "embedding",
        "embed-provider", "https://embed.example/v1", "emb-tok"
    )

    session.commit.assert_awaited_once()


