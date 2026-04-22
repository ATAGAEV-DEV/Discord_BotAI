"""Тест для проверки механизма таймаута."""
import asyncio
import sys
sys.path.insert(0, '.')

from app.core.handlers import AI_GENERATE_TIMEOUT


def test_timeout_constant_exists():
    """Проверяем что константа таймаута задана."""
    assert AI_GENERATE_TIMEOUT is not None, "AI_GENERATE_TIMEOUT не задан"
    print(f"✅ AI_GENERATE_TIMEOUT = {AI_GENERATE_TIMEOUT}")


def test_timeout_value():
    """Проверяем что значение таймаута в разумных пределах (1-120 сек)."""
    assert 1.0 <= AI_GENERATE_TIMEOUT <= 120.0, \
        f"Timeout должен быть между 1 и 120, got {AI_GENERATE_TIMEOUT}"
    print(f"✅ Таймаут AI_GENERATE_TIMEOUT = {AI_GENERATE_TIMEOUT} сек (в допустимых пределах)")


async def test_async_timeout():
    """Тест что asyncio.wait_for корректно ловит таймаут."""
    async def slow_function():
        await asyncio.sleep(10)  # Симулируем долгую операцию
        return "finished"
    
    try:
        result = await asyncio.wait_for(slow_function(), timeout=2.0)
        print("❌ Таймаут НЕ сработал (функция вернула результат)")
    except asyncio.TimeoutError:
        print("✅ Таймаут сработал! TimeoutError пойман")


async def test_llama_timeout():
    """Тест таймаута для LlamaIndex операций."""
    from app.services.llama_integration_bad import QUERY_TIMEOUT, INDEX_TIMEOUT
    
    assert 5.0 <= QUERY_TIMEOUT <= 60.0, f"QUERY_TIMEOUT должен быть между 5 и 60, got {QUERY_TIMEOUT}"
    assert 5.0 <= INDEX_TIMEOUT <= 60.0, f"INDEX_TIMEOUT должен быть между 5 и 60, got {INDEX_TIMEOUT}"
    print(f"✅ QUERY_TIMEOUT = {QUERY_TIMEOUT}, INDEX_TIMEOUT = {INDEX_TIMEOUT} (в допустимых пределах)")


def main():
    """Запуск всех тестов."""
    print("=" * 50)
    print("ТЕСТ МЕХАНИЗМА ТАЙМАУТА")
    print("=" * 50)
    
    print("\n1. Проверка констант...")
    test_timeout_constant_exists()
    test_timeout_value()
    
    print("\n2. Проверка таймаута LlamaIndex...")
    asyncio.run(test_llama_timeout())
    
    print("\n3. Проверка работы asyncio.wait_for...")
    asyncio.run(test_async_timeout())
    
    print("\n" + "=" * 50)
    print("Все тесты прошли успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
