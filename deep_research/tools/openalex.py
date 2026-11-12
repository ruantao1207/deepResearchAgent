"""OpenAlex 检索：免 Key、额度宽松，作为 Semantic Scholar 被限流时的回退数据源。

提供引用数、发表日期、期刊/会议；摘要以倒排索引形式给出，这里还原为文本。
设置环境变量 OPENALEX_MAILTO 可进入 polite pool（更稳定）。
"""

from __future__ import annotations

import os
import re

import httpx

from ..state import Source

OPENALEX_API = "https://api.openalex.org/works"
_SELECT = (
    "id,doi,title,display_name,abstract_inverted_index,publication_year,publication_date,"
    "primary_location,authorships,cited_by_count"
)


def rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    return " ".join(w for _, w in sorted(positions))


def parse_works(data: dict) -> list[Source]:
    out: list[Source] = []
    for w in data.get("results") or []:
        title = (w.get("title") or w.get("display_name") or "").strip()
        if not title:
            continue
        doi = (w.get("doi") or "").replace("https://doi.org/", "").strip().lower()
        m = re.fullmatch(r"10\.48550/arxiv\.(.+)", doi)
        if m:
            key, url = f"arxiv:{m.group(1)}", f"https://arxiv.org/abs/{m.group(1)}"
        elif doi:
            key, url = f"doi:{doi}", f"https://doi.org/{doi}"
        else:
            key, url = f"openalex:{w.get('id')}", w.get("id") or ""
        loc = w.get("primary_location") or {}
        venue = ((loc.get("source") or {}).get("display_name")) or ""
        out.append(
            Source(
                key=key,
                source_type="paper",
                origin="openalex",
                title=title,
                url=url,
                snippet=rebuild_abstract(w.get("abstract_inverted_index")),
                authors=[
                    ((a.get("author") or {}).get("display_name") or "")
                    for a in (w.get("authorships") or [])
                ],
                published=w.get("publication_date") or str(w.get("publication_year") or ""),
                venue=venue,
                citation_count=w.get("cited_by_count") or 0,
            )
        )
    return out


async def search_openalex(query: str, limit: int = 5) -> list[Source]:
    params: dict = {"search": query, "per-page": limit, "select": _SELECT}
    if mail := os.getenv("OPENALEX_MAILTO"):
        params["mailto"] = mail
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(OPENALEX_API, params=params)
        resp.raise_for_status()
    return parse_works(resp.json())
