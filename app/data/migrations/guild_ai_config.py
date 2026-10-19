"""Идемпотентная миграция и явный backfill AI credentials.

Миграция намеренно не вызывается из ``on_ready``. Оператор должен передать
отдельный embedding secret mapping или явно заданный migration token; старый
chat token никогда не является источником embedding credentials.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.core.config import POLZA_PROVIDER
from app.data.models import Base, engine_local, engine_remote


class GuildAIConfigMigrationError(RuntimeError):
    """Безопасная ошибка миграции без вывода значений credentials."""


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Сводка миграции без секретов."""

    guild_count: int
    backfilled_guilds: tuple[int, ...]
    changed: bool


_NEW_COLUMNS: dict[str, str] = {
    "embedding_provider": "VARCHAR(50)",
    "embedding_api_url": "VARCHAR(500)",
    "embedding_token": "VARCHAR(1000)",
}


async def _table_columns(connection: AsyncConnection) -> set[str]:
    """Возвращает колонки таблицы через синхронный SQLAlchemy inspector."""
    return await connection.run_sync(
        lambda sync_connection: {
            column["name"] for column in inspect(sync_connection).get_columns("guild_ai_configs")
        }
    )


async def _prepare_schema(connection: AsyncConnection) -> set[str]:
    """Создаёт отсутствующую таблицу и добавляет новые колонки nullable."""
    await connection.run_sync(Base.metadata.create_all)
    columns = await _table_columns(connection)
    dialect = connection.dialect.name
    for name, sql_type in _NEW_COLUMNS.items():
        if name in columns:
            continue
        if dialect == "postgresql":
            statement = f"ALTER TABLE guild_ai_configs ADD COLUMN IF NOT EXISTS {name} {sql_type}"
        else:
            statement = f"ALTER TABLE guild_ai_configs ADD COLUMN {name} {sql_type}"
        await connection.execute(text(statement))
    return await _table_columns(connection)


def _secret(
    mapping: Mapping[int, str] | None,
    guild_id: int,
    fallback: str | None,
) -> str | None:
    """Выбирает только явно переданный непустой secret."""
    value = mapping.get(guild_id) if mapping is not None else None
    return (
        value.strip()
        if isinstance(value, str) and value.strip()
        else (fallback.strip() if isinstance(fallback, str) and fallback.strip() else None)
    )


def _url(
    mapping: Mapping[int, str] | None,
    guild_id: int,
    fallback: str | None,
) -> str | None:
    """Выбирает только явный URL mapping или migration URL."""
    return _secret(mapping, guild_id, fallback)


async def _backfill(
    connection: AsyncConnection,
    *,
    secret_mapping: Mapping[int, str] | None,
    url_mapping: Mapping[int, str] | None,
    migration_token: str | None,
    migration_url: str | None,
) -> tuple[dict[int, tuple[Any, ...]], set[int]]:
    """Заполняет обе новые колонки и возвращает безопасно сравнимый snapshot."""
    result = await connection.execute(
        text(
            "SELECT guild_id, provider, api_url, token, chat_model, mini_model, "
            "embedding_provider, embedding_api_url, embedding_token, embedding_model "
            "FROM guild_ai_configs ORDER BY guild_id"
        )
    )
    rows = result.mappings().all()
    missing_secret: list[int] = []
    missing_url: list[int] = []
    changed_guilds: set[int] = set()

    for row in rows:
        guild_id = int(row["guild_id"])
        if row["provider"] != POLZA_PROVIDER or (
            isinstance(row["embedding_provider"], str)
            and row["embedding_provider"].strip()
            and row["embedding_provider"] != POLZA_PROVIDER
        ):
            raise GuildAIConfigMigrationError(
                f"Миграция отклонена: неподдерживаемый provider для гильдии {guild_id}"
            )
        token = _secret(secret_mapping, guild_id, migration_token)
        url = _url(url_mapping, guild_id, migration_url)
        if not isinstance(row["embedding_provider"], str) or not row["embedding_provider"].strip():
            await connection.execute(
                text(
                    "UPDATE guild_ai_configs SET embedding_provider = :embedding_provider "
                    "WHERE guild_id = :guild_id"
                ),
                {"embedding_provider": POLZA_PROVIDER, "guild_id": guild_id},
            )
            changed_guilds.add(guild_id)
        if not isinstance(row["embedding_token"], str) or not row["embedding_token"].strip():
            if token is None:
                missing_secret.append(guild_id)
            else:
                await connection.execute(
                    text(
                        "UPDATE guild_ai_configs SET embedding_token = :embedding_token "
                        "WHERE guild_id = :guild_id"
                    ),
                    {"embedding_token": token, "guild_id": guild_id},
                )
                changed_guilds.add(guild_id)
        if not isinstance(row["embedding_api_url"], str) or not row["embedding_api_url"].strip():
            if url is None:
                missing_url.append(guild_id)
            else:
                await connection.execute(
                    text(
                        "UPDATE guild_ai_configs SET embedding_api_url = :embedding_api_url "
                        "WHERE guild_id = :guild_id"
                    ),
                    {"embedding_api_url": url, "guild_id": guild_id},
                )
                changed_guilds.add(guild_id)

    if missing_secret:
        raise GuildAIConfigMigrationError(
            "Миграция остановлена: отсутствует embedding secret для гильдий "
            + ", ".join(map(str, missing_secret))
        )
    if missing_url:
        raise GuildAIConfigMigrationError(
            "Миграция остановлена: отсутствует embedding URL для гильдий "
            + ", ".join(map(str, missing_url))
        )

    final = await connection.execute(
        text(
            "SELECT guild_id, provider, api_url, token, chat_model, mini_model, "
            "embedding_provider, embedding_api_url, embedding_token, embedding_model "
            "FROM guild_ai_configs ORDER BY guild_id"
        )
    )
    snapshot: dict[int, tuple[Any, ...]] = {}
    for row in final.mappings().all():
        guild_id = int(row["guild_id"])
        if row["provider"] != POLZA_PROVIDER or row["embedding_provider"] != POLZA_PROVIDER:
            raise GuildAIConfigMigrationError(
                f"Миграция отклонена: неподдерживаемый provider для гильдии {guild_id}"
            )
        required = (
            row["api_url"],
            row["token"],
            row["chat_model"],
            row["mini_model"],
            row["embedding_api_url"],
            row["embedding_token"],
            row["embedding_model"],
        )
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise GuildAIConfigMigrationError(
                f"Миграция отклонена: неполная конфигурация гильдии {guild_id}"
            )
        snapshot[guild_id] = tuple(row.values())
    return snapshot, changed_guilds


async def _set_not_null(connection: AsyncConnection) -> None:
    """Устанавливает NOT NULL после успешной проверки обеих копий."""
    if connection.dialect.name != "postgresql":
        return
    for column in ("embedding_provider", "embedding_api_url", "embedding_token"):
        await connection.execute(
            text(f"ALTER TABLE guild_ai_configs ALTER COLUMN {column} SET NOT NULL")
        )


async def migrate(
    *,
    secret_mapping: Mapping[int, str] | None = None,
    url_mapping: Mapping[int, str] | None = None,
    migration_token: str | None = None,
    migration_url: str | None = None,
    local: AsyncEngine = engine_local,
    remote: AsyncEngine = engine_remote,
) -> MigrationResult:
    """Выполняет один согласованный backfill в обеих PostgreSQL БД.

    ``migration_token`` и ``migration_url`` должны быть переданы явно. Для
    удобства CLI допускается только отдельная переменная ``AI_EMBEDDING_TOKEN``
    и ``AI_EMBEDDING_API_URL``; ``AI_TOKEN`` намеренно не читается.
    """
    explicit_token = (
        migration_token if migration_token is not None else os.getenv("AI_EMBEDDING_TOKEN")
    )
    explicit_url = migration_url if migration_url is not None else os.getenv("AI_EMBEDDING_API_URL")
    local_connection = await local.connect()
    remote_connection = await remote.connect()
    local_transaction = await local_connection.begin()
    remote_transaction = await remote_connection.begin()
    try:
        await _prepare_schema(local_connection)
        await _prepare_schema(remote_connection)
        local_snapshot, local_changed = await _backfill(
            local_connection,
            secret_mapping=secret_mapping,
            url_mapping=url_mapping,
            migration_token=explicit_token,
            migration_url=explicit_url,
        )
        remote_snapshot, remote_changed = await _backfill(
            remote_connection,
            secret_mapping=secret_mapping,
            url_mapping=url_mapping,
            migration_token=explicit_token,
            migration_url=explicit_url,
        )
        if local_snapshot != remote_snapshot:
            raise GuildAIConfigMigrationError("Миграция остановлена: копии БД не совпадают")
        if local_changed != remote_changed:
            raise GuildAIConfigMigrationError(
                "Миграция остановлена: результаты backfill не совпадают"
            )
        await _set_not_null(local_connection)
        await _set_not_null(remote_connection)
        await local_transaction.commit()
        await remote_transaction.commit()
    except GuildAIConfigMigrationError:
        await local_transaction.rollback()
        await remote_transaction.rollback()
        raise
    except Exception:
        await local_transaction.rollback()
        await remote_transaction.rollback()
        raise GuildAIConfigMigrationError("Миграция AI-конфигурации не выполнена") from None
    finally:
        await local_connection.close()
        await remote_connection.close()

    changed_guilds = tuple(sorted(local_changed))
    if changed_guilds:
        from app.core.ai_config import invalidate_embedding_runtime
        from app.data import guild_ai_config_cache

        for guild_id in changed_guilds:
            guild_ai_config_cache.invalidate(guild_id)
            invalidate_embedding_runtime(guild_id)
    return MigrationResult(
        guild_count=len(local_snapshot),
        backfilled_guilds=changed_guilds,
        changed=bool(changed_guilds),
    )
