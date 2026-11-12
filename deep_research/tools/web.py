"""网页 / 业界动态检索：优先 Tavily，未配置时回退 DuckDuckGo。"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse

import httpx

from ..state import Source

TAVILY_API = "https://api.tavily.com/search"


def _site(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


async def _tavily(query: str, limit: int, api_key: str) -> list[Source]:
    payload = {
        "query": query,
        "max_results": limit,
        "search_depth": "advanced",
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            TAVILY_API, json=payload, headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        data = resp.json()
    out: list[Source] = []
    for r in data.get("results", []):
        url = r.get("url", "")
        out.append(
            Source(
                key=f"url:{url.rstrip('/')}",
                source_type="web",
                origin="tavily",
                title=r.get("title") or url,
                url=url,
                snippet=(r.get("content") or "")[:2000],
                published=r.get("published_date") or "",
                venue=_site(url),
            )
        )
    return out


def _ddg_sync(query: str, limit: int) -> list[Source]:
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover - 旧包名
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "未配置 TAVILY_API_KEY，且未安装 ddgs（pip install ddgs），无法进行网页检索"
            ) from e
    out: list[Source] = []
    for r in DDGS().text(query, max_results=limit) or []:
        url = r.get("href", "")
        out.append(
            Source(
                key=f"url:{url.rstrip('/')}",
                source_type="web",
                origin="duckduckgo",
                title=r.get("title") or url,
                url=url,
                snippet=r.get("body", ""),
                venue=_site(url),
            )
        )
    return out


async def search_web(query: str, limit: int = 5) -> list[Source]:
    if key := os.getenv("TAVILY_API_KEY"):
        return await _tavily(query, limit, key)
    return await asyncio.to_thread(_ddg_sync, query, limit)
