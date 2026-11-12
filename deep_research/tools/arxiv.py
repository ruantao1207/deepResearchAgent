"""arXiv 检索（官方 Atom API，无需 Key）。"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET

import httpx

from ..state import Source
from ._utils import LoopLocal

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_STOPWORDS = {"a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with", "via", "by"}

# arXiv 要求低频访问（约 3 秒 1 次），这里串行化请求
_lock = LoopLocal(1)
_MIN_INTERVAL = 3.0


def build_query(query: str) -> str:
    tokens = [t for t in re.findall(r"[A-Za-z0-9][\w\-\.]*", query) if t.lower() not in _STOPWORDS]
    if not tokens:
        return f'all:"{query}"'
    return " AND ".join(f"all:{t}" for t in tokens[:6])


def parse_atom(xml_text: str) -> list[Source]:
    root = ET.fromstring(xml_text)
    out: list[Source] = []
    for entry in root.findall("a:entry", _NS):
        raw_id = (entry.findtext("a:id", "", _NS) or "").strip()
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        if not arxiv_id:
            continue
        title = " ".join((entry.findtext("a:title", "", _NS) or "").split())
        summary = " ".join((entry.findtext("a:summary", "", _NS) or "").split())
        authors = [
            (a.findtext("a:name", "", _NS) or "").strip() for a in entry.findall("a:author", _NS)
        ]
        cat = entry.find("arxiv:primary_category", _NS)
        out.append(
            Source(
                key=f"arxiv:{arxiv_id.lower()}",
                source_type="paper",
                origin="arxiv",
                title=title,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                snippet=summary,
                authors=authors,
                published=(entry.findtext("a:published", "", _NS) or "")[:10],
                venue=f"arXiv ({cat.get('term')})" if cat is not None else "arXiv",
            )
        )
    return out


async def search_arxiv(query: str, limit: int = 5) -> list[Source]:
    params = {
        "search_query": build_query(query),
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
    }
    async with _lock.get():
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for attempt in range(3):
                resp = await client.get(ARXIV_API, params=params)
                if resp.status_code in (429, 503):
                    await asyncio.sleep(_MIN_INTERVAL * (attempt + 1))
                    continue
                resp.raise_for_status()
                break
            else:
                resp.raise_for_status()
        await asyncio.sleep(_MIN_INTERVAL)
    return parse_atom(resp.text)
