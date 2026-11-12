"""Semantic Scholar 检索：补充引用数、会议/期刊信息，覆盖非 arXiv 论文。

限流策略（无 Key 时共享额度极易 429）：
- 全局串行 + 请求间隔 ≥ 1s，并遵守 Retry-After
- 重试仍 429 则熔断 COOLDOWN 秒，期间直接走回退，避免每个子问题都白等
- 被限流/网络失败时自动回退到 OpenAlex，保证论文检索不中断
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from ..state import Source
from ._utils import LoopLocal
from .openalex import search_openalex

logger = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,abstract,url,year,venue,authors,citationCount,externalIds,publicationDate,tldr"

COOLDOWN = 90.0  # 熔断时长（秒）
MAX_RETRIES = 2
MIN_INTERVAL = 1.1

_lock = LoopLocal(1)
_blocked_until = 0.0


def parse_papers(data: dict) -> list[Source]:
    out: list[Source] = []
    for p in data.get("data") or []:
        title = (p.get("title") or "").strip()
        if not title:
            continue
        ext = p.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        abstract = p.get("abstract") or ((p.get("tldr") or {}).get("text")) or ""
        out.append(
            Source(
                key=f"arxiv:{arxiv_id.lower()}" if arxiv_id else f"s2:{p.get('paperId')}",
                source_type="paper",
                origin="semantic_scholar",
                title=title,
                url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (p.get("url") or ""),
                snippet=abstract,
                authors=[a.get("name", "") for a in (p.get("authors") or [])],
                published=p.get("publicationDate") or str(p.get("year") or ""),
                venue=p.get("venue") or "",
                citation_count=p.get("citationCount") or 0,
            )
        )
    return out


class RateLimited(Exception):
    pass


async def _fetch_s2(query: str, limit: int) -> list[Source]:
    global _blocked_until
    headers = {}
    if key := os.getenv("S2_API_KEY"):
        headers["x-api-key"] = key
    params = {"query": query, "limit": limit, "fields": _FIELDS}

    async with _lock.get():
        if time.monotonic() < _blocked_until:
            raise RateLimited("Semantic Scholar 熔断冷却中")
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            delay = 2.0
            for attempt in range(MAX_RETRIES + 1):
                resp = await client.get(S2_API, params=params)
                if resp.status_code == 429:
                    if attempt == MAX_RETRIES:
                        _blocked_until = time.monotonic() + COOLDOWN
                        raise RateLimited("Semantic Scholar 429 限流")
                    retry_after = resp.headers.get("Retry-After", "")
                    wait = min(float(retry_after), 8.0) if retry_after.isdigit() else delay
                    await asyncio.sleep(wait)
                    delay *= 2
                    continue
                resp.raise_for_status()
                break
        await asyncio.sleep(MIN_INTERVAL)  # 保持 ≤ 1 req/s
    return parse_papers(resp.json())


async def search_semantic_scholar(query: str, limit: int = 5) -> list[Source]:
    try:
        return await _fetch_s2(query, limit)
    except (RateLimited, httpx.HTTPError) as e:
        logger.info("Semantic Scholar 不可用（%s），回退 OpenAlex: %s", e, query)
        return await search_openalex(query, limit)
