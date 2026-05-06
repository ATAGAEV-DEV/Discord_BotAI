import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.data.models import async_session

DB_TIMEOUT = 15


def db_operation(operation_name: str) -> Callable:
    """Декоратор для устранения бойлерплейта БД-операций.

    Оборачивает функцию в async with session + try/except с таймаутом.
    Декорируемая функция должна принимать session: AsyncSession первым аргументом.
    """

    def decorator(func_inner: Callable) -> Callable:
        @wraps(func_inner)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with async_session() as session:
                try:
                    return await asyncio.wait_for(
                        func_inner(session, *args, **kwargs), timeout=DB_TIMEOUT
                    )
                except TimeoutError as e:
                    raise TimeoutError(f"Таймаут при {operation_name}.") from e
                except OperationalError as e:
                    raise TimeoutError(f"Ошибка подключения/таймаут БД при {operation_name}.") from e
                except SQLAlchemyError as e:
                    raise RuntimeError(f"Ошибка базы данных при {operation_name}: {e}") from e
                except Exception as e:
                    raise RuntimeError(f"Непредвиденная ошибка при {operation_name}: {e}") from e

        return wrapper

    return decorator
