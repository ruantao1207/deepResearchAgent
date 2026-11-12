"""GitHub 仓库检索：开源项目与代码实现（GITHUB_TOKEN 可选，用于提高限额）。"""

from __future__ import annotations

import os

import httpx

from ..state import Source

GITHUB_API = "https://api.github.com/search/repositories"


def _fmt_stars(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def parse_repos(data: dict) -> list[Source]:
    out: list[Source] = []
    for r in data.get("items") or []:
        url = r.get("html_url", "")
        if not url:
            continue
        stars = r.get("stargazers_count") or 0
        bits = [r.get("description") or ""]
        if r.get("topics"):
            bits.append("topics: " + ", ".join(r["topics"][:8]))
        if r.get("language"):
            bits.append(f"language: {r['language']}")
        bits.append(f"stars: {stars}, forks: {r.get('forks_count', 0)}")
        out.append(
            Source(
                key=f"url:{url.rstrip('/').lower()}",
                source_type="web",
                origin="github",
                title=r.get("full_name") or url,
                url=url,
                snippet=". ".join(b for b in bits if b),
                published=(r.get("pushed_at") or r.get("created_at") or "")[:10],
                venue=f"GitHub ★{_fmt_stars(stars)}",
            )
        )
    return out


async def search_github(query: str, limit: int = 5) -> list[Source]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "deep-research-agent"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": limit}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        resp = await client.get(GITHUB_API, params=params)
        if resp.status_code in (403, 429):
            raise RuntimeError("GitHub API 限流（可在设置中配置 GITHUB_TOKEN 提高限额）")
        resp.raise_for_status()
    return parse_repos(resp.json())
