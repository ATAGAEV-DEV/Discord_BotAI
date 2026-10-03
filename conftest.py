"""Конфигурация pytest: интеграционные БД доступны только при явном запуске."""

import os
import sys
import warnings
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import event
from sqlalchemy.engine import Engine

from tests.postgres_support import guard_connection, install_test_environment, load_test_urls

TEST_URLS = pytest.StashKey[dict]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Добавляет явное разрешение интеграционных тестов."""
    parser.addoption(
        "--run-integration", action="store_true", help="Только PostgreSQL integration-тесты"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Устанавливает тестовое окружение до collection и импорта моделей."""
    event.listen(Engine, "do_connect", guard_connection)
    if config.getoption("--run-integration"):
        if os.getenv("PYTEST_XDIST_WORKER") or getattr(config.option, "numprocesses", 0):
            raise pytest.UsageError("Интеграционные проверки запускаются только последовательно")
        if "app.data.models" in sys.modules:
            raise pytest.UsageError("Модели импортированы до настройки тестовых БД")
        try:
            urls = load_test_urls(Path(__file__).parent / ".env.test", os.environ)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from None
        install_test_environment(urls)
        config.stash[TEST_URLS] = urls
        if any(url.username == "postgres" for url in urls.values()):
            warnings.warn(
                "Тесты используют postgres; рекомендуется отдельная роль bot_test "
                "с правами только на test_atagaev.",
                pytest.PytestWarning,
                stacklevel=1,
            )
    else:
        # Unit-тестам нужны лишь валидные URL при импорте, но не реальные БД.
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://unit:unit@127.0.0.1:1/unit"
        os.environ["DATABASE_URL_LOCAL"] = "postgresql+asyncpg://unit:unit@127.0.0.1:2/unit"
    load_dotenv(override=False)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """По умолчанию даже не импортирует интеграционные тесты."""
    tests_path = Path(__file__).parent / "tests"
    if collection_path == tests_path / "integration":
        return not config.getoption("--run-integration")
    if (
        config.getoption("--run-integration")
        and collection_path.is_relative_to(tests_path)
        and not collection_path.is_relative_to(tests_path / "integration")
        and collection_path.name.startswith("test_")
        and collection_path.suffix == ".py"
    ):
        return True
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Разделяет unit и integration даже при явном указании отдельного файла."""
    selected, deselected = [], []
    enabled = config.getoption("--run-integration")
    for item in items:
        integration = "integration" in item.path.relative_to(config.rootpath).parts
        if integration:
            item.add_marker(pytest.mark.integration)
        integration = item.get_closest_marker("integration") is not None
        (selected if integration == enabled else deselected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Снимает глобальный запрет подключений при завершении pytest."""
    if event.contains(Engine, "do_connect", guard_connection):
        event.remove(Engine, "do_connect", guard_connection)
