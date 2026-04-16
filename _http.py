"""
_http.py
Shared aiohttp.ClientSession — one per process, managed via FastAPI lifespan.
Import get_session() in all async fetchers.
"""
import aiohttp
from typing import Optional

_session: Optional[aiohttp.ClientSession] = None


def get_session() -> aiohttp.ClientSession:
    if _session is None or _session.closed:
        raise RuntimeError(
            "aiohttp session not initialized — "
            "call await init_session() in the FastAPI lifespan first."
        )
    return _session


async def init_session() -> None:
    global _session
    timeout = aiohttp.ClientTimeout(total=15, connect=5)
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=4)
    _session = aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={"User-Agent": "WeatherArb/3.0 (contact@example.com)"},
    )


async def close_session() -> None:
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None
