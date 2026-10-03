"""Проверки предохранителей без сетевых подключений и настоящих секретов."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from tests.postgres_support import (
    ACTIVE_SCHEMAS,
    guard_connection,
    install_test_environment,
    load_test_urls,
    validate_test_url,
)

REMOTE = "postgresql+asyncpg://bot_test:fake@91.205.220.24:5432/test_atagaev"
LOCAL = "postgresql+asyncpg://bot_test:fake@192.168.0.147:5433/test_atagaev"
DIALECT = SimpleNamespace(name="postgresql", driver="asyncpg")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-a-url-secret",
        REMOTE.replace("5432", "bad-port"),
        REMOTE.replace("test_atagaev", "atagaev"),
        REMOTE.replace("91.205.220.24", "127.0.0.1"),
        REMOTE.replace("5432", "5433"),
        REMOTE.replace("+asyncpg", ""),
        REMOTE.replace(":fake", ""),
        REMOTE + "?host=another-server",
        LOCAL,
    ],
)
def test_unsafe_urls_are_rejected_without_secrets(value: str | None) -> None:
    """Отсутствующий, рабочий или подменённый адрес не попадает в движок или вывод."""
    with pytest.raises(ValueError, match="TEST_DATABASE_URL") as error:
        validate_test_url(value, "TEST_DATABASE_URL")
    assert "fake" not in str(error.value)
    assert "not-a-url-secret" not in str(error.value)
    assert "postgresql" not in str(error.value)


def test_environment_overrides_file_and_production_values(tmp_path: Path) -> None:
    """Использует только TEST_*; пустое значение окружения тоже не заменяет файлом."""
    path = tmp_path / "test-settings"
    path.write_text(f"TEST_DATABASE_URL={REMOTE}\nTEST_DATABASE_URL_LOCAL={LOCAL}\n")
    urls = load_test_urls(path, {"TEST_DATABASE_URL": REMOTE.replace("fake", "override")})
    assert urls["TEST_DATABASE_URL"].password == "override"
    assert urls["TEST_DATABASE_URL_LOCAL"].host == "192.168.0.147"
    with pytest.raises(ValueError):
        load_test_urls(path, {"TEST_DATABASE_URL": ""})
    with pytest.raises(ValueError):
        load_test_urls(tmp_path / "absent", {"DATABASE_URL": REMOTE, "DATABASE_URL_LOCAL": LOCAL})


def test_dotenv_does_not_interpolate_passwords(tmp_path: Path) -> None:
    """Спецсимволы пароля не заменяются значениями окружения."""
    path = tmp_path / "test-settings"
    path.write_text(
        f"TEST_DATABASE_URL={REMOTE.replace('fake', '${SECRET}')}\nTEST_DATABASE_URL_LOCAL={LOCAL}"
    )
    assert load_test_urls(path, {})["TEST_DATABASE_URL"].password == "${SECRET}"


def test_test_environment_replaces_both_application_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Приложение получает только заранее проверенные адреса до импорта моделей."""
    for key in ("DATABASE_URL", "DATABASE_URL_LOCAL", "DATABASE_SCHEMA"):
        monkeypatch.setenv(key, "production-placeholder")
    urls = load_test_urls(
        tmp_path / "absent", {"TEST_DATABASE_URL": REMOTE, "TEST_DATABASE_URL_LOCAL": LOCAL}
    )
    install_test_environment(urls)
    assert os.environ["DATABASE_URL"] == REMOTE
    assert os.environ["DATABASE_URL_LOCAL"] == LOCAL
    assert os.environ["DATABASE_SCHEMA"] == "pytest_unconfigured"


def test_connection_guard_requires_active_schema() -> None:
    """Даже тестовая база недоступна без собственной активной схемы."""
    params = {
        "host": "91.205.220.24", "port": 5432, "database": "test_atagaev",
        "user": "bot_test", "password": "fake", "server_settings": {"search_path": "public"},
    }
    with pytest.raises(RuntimeError, match="запрещено"):
        guard_connection(DIALECT, None, [], params)
    schema = "pytest_" + "0" * 32
    ACTIVE_SCHEMAS.add(schema)
    try:
        params["server_settings"]["search_path"] = schema
        guard_connection(DIALECT, None, [], params)
        params["database"] = "atagaev"
        with pytest.raises(RuntimeError, match="запрещено"):
            guard_connection(DIALECT, None, [], params)
    finally:
        ACTIVE_SCHEMAS.remove(schema)


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "production"},
        {"port": 5433},
        {"password": ""},
        {"dsn": "postgresql://production"},
        {"server_settings": {"search_path": "public"}},
        {"server_settings": {"search_path": "pytest_" + "0" * 32 + ",public"}},
    ],
)
def test_guard_rejects_driver_parameter_overrides(overrides: dict) -> None:
    """connect_args не может обойти URL-валидацию или расширить search_path."""
    schema = "pytest_" + "0" * 32
    params = {
        "host": "91.205.220.24", "port": 5432, "database": "test_atagaev",
        "user": "bot_test", "password": "fake", "server_settings": {"search_path": schema},
    }
    ACTIVE_SCHEMAS.add(schema)
    try:
        with pytest.raises(RuntimeError, match="запрещено"):
            guard_connection(DIALECT, None, [], params | overrides)
    finally:
        ACTIVE_SCHEMAS.remove(schema)


async def test_unit_run_blocks_actual_engine_before_network() -> None:
    """Глобальный SQLAlchemy hook срабатывает раньше драйверного connect."""
    engine = create_async_engine(REMOTE)
    try:
        with pytest.raises(RuntimeError, match="запрещено"):
            async with engine.connect():
                pytest.fail("Соединение не должно открываться")
    finally:
        await engine.dispose()


@pytest.mark.parametrize("worker", [False, True])
def test_pytest_refuses_unsafe_configuration_before_collection(worker: bool) -> None:
    """CLI отказывает до collection: некорректный URL или worker не подключается к БД."""
    root = Path(__file__).resolve().parents[1]
    environ = dict(os.environ, TEST_DATABASE_URL="invalid-secret", TEST_DATABASE_URL_LOCAL=LOCAL)
    environ["PYTHONIOENCODING"] = "utf-8"
    if worker:
        environ["PYTEST_XDIST_WORKER"] = "gw0"
    else:
        environ.pop("PYTEST_XDIST_WORKER", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--run-integration", "--collect-only", "-q",
         str(root / "tests" / "integration" / "test_dual_database.py")],
        cwd=root, env=environ, capture_output=True, text=True, encoding="utf-8", timeout=20,
        check=False,
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "invalid-secret" not in result.stderr
    assert "collected" not in result.stdout
    assert ("последовательно" if worker else "TEST_DATABASE_URL") in result.stderr