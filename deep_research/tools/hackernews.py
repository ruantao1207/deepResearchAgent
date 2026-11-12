"""Hacker News 检索（Algolia API，免 Key）：业界讨论与舆情。"""

from __future__ import annotations

import re

import httpx

from ..state import Source

HN_API = "https://hn.algolia.com/api/v1/search"


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def parse_hits(data: dict) -> list[Source]:
    out: list[Source] = []
    for h in data.get("hits") or []:
        title = (h.get("title") or h.get("story_title") or "").strip()
        oid = h.get("objectID")
        if not title or not oid:
            continue
        discussion = f"https://news.ycombinator.com/item?id={oid}"
        url = h.get("url") or discussion
        points, comments = h.get("points") or 0, h.get("num_comments") or 0
        snippet = f"Hacker News 讨论：{points} points，{comments} comments。" + _strip_html(h.get("story_text") or "")
        out.append(
            Source(
                key=f"url:{url.rstrip('/').lower()}",
                source_type="web",
                origin="hackernews",
                title=title,
                url=url,
                snippet=snippet[:1500],
                published=(h.get("created_at") or "")[:10],
                venue=f"Hacker News ▲{points}",
            )
        )
    return out


async def search_hackernews(query: str, limit: int = 5) -> list[Source]:
    params = {"query": query, "tags": "story", "hitsPerPage": limit}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(HN_API, params=params)
        resp.raise_for_status()
    return parse_hits(resp.json())
