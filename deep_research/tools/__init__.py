"""检索工具：arXiv / Semantic Scholar / Web（Tavily 或 DuckDuckGo）。

所有工具统一返回 list[Source]，并发调用由 `search_all` 负责。
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from ..state import Source
from .arxiv import search_arxiv
from .github import search_github
from .hackernews import search_hackernews
from .huggingface import search_huggingface
from .semantic_scholar import search_semantic_scholar
from .web import search_web

logger = logging.getLogger(__name__)

SEARCHERS = {
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "web": search_web,
    "github": search_github,
    "huggingface": search_huggingface,
    "hackernews": search_hackernews,
}

RETRIES = 2  # 失败后额外重试次数
RETRY_DELAY = 1.5


async def _call_with_retry(fn, query: str, limit: int) -> list[Source]:
    """瞬时错误（网络/超时/5xx/搜索引擎临时拒绝）自动重试；配置类错误与 4xx 不重试。"""
    last: BaseException | None = None
    for attempt in range(RETRIES + 1):
        try:
            return await fn(query, limit)
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise
            last = e
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
        if attempt < RETRIES:
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    assert last is not None
    raise last


async def search_all(
    queries: list[str], sources: list[str], limit: int
) -> tuple[list[Source], list[str]]:
    """对 (query × source) 并发检索，返回 (按数据源交错排列的结果, 错误信息)。"""
    jobs, labels = [], []
    for src in sources:
        fn = SEARCHERS.get(src)
        if fn is None:
            continue
        for q in queries:
            jobs.append(_call_with_retry(fn, q, limit))
            labels.append(f"{src}:{q}")

    results = await asyncio.gather(*jobs, return_exceptions=True)
    buckets: list[list[Source]] = []
    errors: list[str] = []
    for label, res in zip(labels, results):
        if isinstance(res, BaseException):
            logger.warning("search failed %s: %r", label, res)
            errors.append(f"{label} -> {type(res).__name__}: {res}")
        else:
            buckets.append(res)
    return _interleave(buckets), errors


def _interleave(buckets: list[list[Source]]) -> list[Source]:
    """轮流从每个结果列表取一条，避免候选被单一数据源霸占。"""
    out: list[Source] = []
    i = 0
    while any(i < len(b) for b in buckets):
        for b in buckets:
            if i < len(b):
                out.append(b[i])
        i += 1
    return out


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", (title or "").lower())


def dedupe_sources(sources: list[Source]) -> list[Source]:
    """按 key / 标题去重；重复项合并补全字段（如 S2 的引用数补到 arXiv 条目上）。"""
    merged: dict[str, Source] = {}
    alias: dict[str, str] = {}
    for s in sources:
        ids = [s.get("key", "")]
        if s.get("source_type") == "paper":
            ids.append("t:" + normalize_title(s.get("title", "")))
        ids = [i for i in ids if i and i != "t:"]
        hit = next((alias[i] for i in ids if i in alias), None)
        if hit is None:
            primary = ids[0] if ids else f"anon:{len(merged)}"
            merged[primary] = dict(s)  # type: ignore[assignment]
            for i in ids:
                alias[i] = primary
        else:
            target = merged[hit]
            for k, v in s.items():
                if v and not target.get(k):
                    target[k] = v  # type: ignore[literal-required]
            if len(s.get("snippet", "")) > len(target.get("snippet", "")):
                target["snippet"] = s["snippet"]
            for i in ids:
                alias.setdefault(i, hit)
    return list(merged.values())


__all__ = [
    "search_all",
    "dedupe_sources",
    "search_arxiv",
    "search_semantic_scholar",
    "search_web",
    "search_github",
    "search_hackernews",
    "search_huggingface",
]
