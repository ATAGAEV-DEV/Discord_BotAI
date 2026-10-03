"""Безопасная конфигурация PostgreSQL-тестов без импорта приложения."""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

ENDPOINTS = {
    "TEST_DATABASE_URL": ("91.205.220.24", 5432),
    "TEST_DATABASE_URL_LOCAL": ("192.168.0.147", 5433),
}
ACTIVE_SCHEMAS: set[str] = set()


def validate_test_url(value: str | None, variable: str) -> URL:
    """Проверяет адрес без вывода URL или секретов даже при ошибке парсинга."""
    error = f"{variable}: требуется разрешённый PostgreSQL endpoint и база test_atagaev"
    try:
        url = make_url(value or "")
        valid = (
            url.drivername == "postgresql+asyncpg"
            and (url.host, url.port) == ENDPOINTS[variable]
            and url.database == "test_atagaev"
            and bool(url.username)
            and bool(url.password)
            and not url.query
        )
    except (ArgumentError, ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise ValueError(error) from None
    return url


def load_test_urls(path: Path, environ: Mapping[str, str]) -> dict[str, URL]:
    """Загружает только тестовые URL; окружение приоритетнее файла, fallback отсутствует."""
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    urls = {
        key: validate_test_url(environ[key] if key in environ else values.get(key), key)
        for key in ENDPOINTS
    }
    if urls["TEST_DATABASE_URL"] == urls["TEST_DATABASE_URL_LOCAL"]:
        raise ValueError("Тестовые подключения должны отличаться")
    return urls


def install_test_environment(urls: dict[str, URL]) -> None:
    """Перенаправляет импортируемые модели, не меняя рабочий .env."""
    for key, url in urls.items():
        os.environ[key.removeprefix("TEST_")] = url.render_as_string(hide_password=False)
    os.environ["DATABASE_SCHEMA"] = "pytest_unconfigured"


def guard_connection(
    dialect: Any, connection_record: Any, args: Any, params: dict[str, Any]
) -> None:
    """Блокирует сеть до подключения: разрешены лишь схемы активных DB-фикстур."""
    schema = params.get("server_settings", {}).get("search_path")
    if (
        dialect.name != "postgresql"
        or dialect.driver != "asyncpg"
        or args
        or "dsn" in params
        or not params.get("user")
        or not params.get("password")
        or params.get("database") != "test_atagaev"
        or (params.get("host"), params.get("port")) not in ENDPOINTS.values()
        or not isinstance(schema, str)
        or not re.fullmatch(r"pytest_[0-9a-f]{32}", schema)
        or schema not in ACTIVE_SCHEMAS
    ):
        raise RuntimeError("pytest: подключение вне изолированной тестовой схемы запрещено")