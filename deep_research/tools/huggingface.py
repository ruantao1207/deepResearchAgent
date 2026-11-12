"""Hugging Face Papers 检索：追踪近期热门论文（带 upvotes），免 Key。

遵循 HF 标准环境变量 HF_ENDPOINT（如 https://hf-mirror.com）；
当国内网络无法连接 huggingface.co 时，自动回退到 OpenAlex 按近期高热度论文检索，避免报错中断。
"""

from __future__ import annotations

import logging
import os

import httpx

from ..state import Source
from .openalex import search_openalex

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"


def parse_papers(data) -> list[Source]:
    out: list[Source] = []
    items = data if isinstance(data, list) else (data.get("papers") or data.get("results") or [])
    for it in items:
        p = it.get("paper") or it
        pid = (p.get("id") or "").strip()
        title = " ".join((p.get("title") or it.get("title") or "").split())
        if not pid or not title:
            continue
        upvotes = p.get("upvotes") or 0
        out.append(
            Source(
                key=f"arxiv:{pid.lower()}",
                source_type="paper",
                origin="huggingface",
                title=title,
                url=f"https://arxiv.org/abs/{pid}",
                snippet=" ".join((p.get("summary") or it.get("summary") or "").split()),
                authors=[a.get("name", "") for a in (p.get("authors") or []) if isinstance(a, dict)],
                published=(p.get("publishedAt") or it.get("publishedAt") or "")[:10],
                venue=f"Hugging Face Papers ▲{upvotes}" if upvotes else "Hugging Face Papers",
            )
        )
    return out


async def search_huggingface(query: str, limit: int = 5) -> list[Source]:
    endpoint = (os.getenv("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(f"{endpoint}/api/papers/search", params={"q": query, "limit": limit})
            resp.raise_for_status()
        return parse_papers(resp.json())[:limit]
    except (httpx.HTTPError, ValueError) as e:
        logger.info("Hugging Face Papers 不可用（%s），回退 OpenAlex: %s", type(e).__name__, query)
        return await search_openalex(query, limit)
