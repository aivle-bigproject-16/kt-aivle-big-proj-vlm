import time
import logging
import functools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def log_duration(func):
    """함수 실행 시간을 INFO 레벨로 로깅하는 데코레이터. 동기/비동기 모두 지원."""
    if _is_async(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("[%s] %.3fs", func.__qualname__, elapsed)
            return result
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("[%s] %.3fs", func.__qualname__, elapsed)
            return result
        return sync_wrapper


def _is_async(func) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(func)
