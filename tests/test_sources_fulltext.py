import asyncio

import pytest

from deep_research.tools import fulltext as ft
from deep_research.tools.github import parse_repos
from deep_research.tools.hackernews import parse_hits
from deep_research.tools.huggingface import parse_papers
from deep_research.nodes import _rank_key

HTML = """<html><head><title>t</title><style>.a{}</style></head><body>
<nav>菜单 首页 关于 联系</nav>
<h1>Paper Title</h1>
<p>Abstract: """ + "We study speculative decoding for large language models and propose a new method. " * 3 + """</p>
<h2>1 Introduction</h2><p>""" + "Introduction text about the motivation and background of the problem. " * 4 + """</p>
<h2>2 Related Work</h2><p>""" + "Related work that should be dropped from the excerpt entirely. " * 4 + """</p>
<h2>3 Experiments</h2><p>""" + "Experiments show a 2.5x speedup on MT-Bench with identical outputs. " * 4 + """</p>
<h2>References</h2><p>""" + "[1] Someone. A reference entry that must not appear in excerpt. " * 4 + """</p>
<script>var x = 1;</script><footer>版权所有 footer footer footer footer footer</footer>
</body></html>"""


def test_html_to_text_and_excerpt():
    text = ft.html_to_text(HTML)
    assert "菜单" not in text and "var x" not in text and "版权所有" not in text
    assert "## Paper Title" in text
    ex = ft.select_excerpt(text, 1200)
    assert "speculative decoding" in ex and "2.5x speedup" in ex
    assert "Related work" not in ex and "reference entry" not in ex  # 丢弃相关工作/参考文献
    assert len(ex) <= 1600


def test_excerpt_prioritizes_key_sections_under_small_budget():
    ex = ft.select_excerpt(ft.html_to_text(HTML), 700)
    assert "2.5x speedup" in ex  # 实验章节优先于篇幅靠前的次要内容


def test_excerpt_plain_page_and_reference_tail():
    page = "\n".join(f"这是一段足够长的网页正文内容，第{i}行，用来测试节选逻辑是否正确。" for i in range(40))
    assert ft.select_excerpt(page, 300).startswith("这是一段足够长")
    pdf_like = ("Body paragraph that is long enough to be kept in the excerpt. " * 30) + "\nReferences\n[1] ref ref ref ref ref"
    assert "[1] ref" not in ft.select_excerpt(pdf_like, 5000)
    assert ft.select_excerpt("   ", 100) == ""


def test_candidate_urls():
    urls = ft.candidate_urls({"key": "arxiv:2401.00001", "url": "https://arxiv.org/abs/2401.00001"})
    assert urls[0] == "https://arxiv.org/html/2401.00001" and urls[-1].endswith("/pdf/2401.00001")
    assert ft.candidate_urls({"key": "url:x", "url": "https://a.com/p"}) == ["https://a.com/p"]
    assert ft.candidate_urls({"key": "url:x"}) == []


@pytest.mark.parametrize(
    "url", ["http://localhost/x", "http://127.0.0.1:8000/", "http://169.254.169.254/latest", "http://10.0.0.5/", "ftp://example.com/x", "file:///etc/passwd", "http:///nohost"]
)
def test_ssrf_guard_rejects_non_public(url):
    assert asyncio.run(ft.is_public_url(url)) is False


def test_fetch_fulltext_rejects_private_without_network():
    src = {"key": "url:x", "url": "http://127.0.0.1:9/secret"}
    assert asyncio.run(ft.fetch_fulltext(src, 1000, timeout=2)) == ""


def test_fake_ip_dns_range_is_allowed_but_private_is_not(monkeypatch):
    import ipaddress

    pub = lambda s: ft._ip_is_public(ipaddress.ip_address(s))
    assert pub("198.18.13.0") and pub("198.19.255.1")  # Clash/Surge fake-ip 网段
    assert pub("8.8.8.8") and pub("2606:4700::1111")
    assert not pub("10.0.0.1") and not pub("192.168.1.1") and not pub("127.0.0.1") and not pub("169.254.169.254") and not pub("::1")
    monkeypatch.setenv("FULLTEXT_ALLOWED_NETWORKS", "100.64.0.0/10, not-a-cidr")
    assert pub("100.64.1.1") and not pub("10.0.0.1")


def test_license_boilerplate_dropped():
    text = "License: arXiv.org perpetual non-exclusive license\n" + "Provided proper attribution is provided, Google hereby grants permission to reproduce.\n" + "## Intro\n" + "Real introduction content that is long enough to be kept in the excerpt. " * 3
    ex = ft.select_excerpt(text, 800)
    assert "License:" not in ex and "Provided proper" not in ex and "Real introduction" in ex


def test_fetch_fulltext_uses_html_then_fallback(monkeypatch):
    calls = []

    async def fake_download(client, url):
        calls.append(url)
        if url.startswith("https://arxiv.org/html"):
            return None  # 新版 HTML 不可用
        return HTML.encode(), "text/html; charset=utf-8", url

    monkeypatch.setattr(ft, "_download", fake_download)
    got = asyncio.run(ft.fetch_fulltext({"key": "arxiv:2401.00001", "url": ""}, 1500))
    assert "2.5x speedup" in got
    assert calls == ["https://arxiv.org/html/2401.00001", "https://ar5iv.labs.arxiv.org/html/2401.00001"]


def test_parse_github():
    [r] = parse_repos(
        {"items": [{"html_url": "https://github.com/a/b", "full_name": "a/b", "description": "desc", "topics": ["llm"],
                    "language": "Python", "stargazers_count": 12345, "forks_count": 7, "pushed_at": "2026-05-01T00:00:00Z"},
                   {"full_name": "no-url"}]}
    )
    assert r["origin"] == "github" and r["venue"] == "GitHub ★12.3k" and r["published"] == "2026-05-01"
    assert "stars: 12345" in r["snippet"] and "topics: llm" in r["snippet"]


def test_parse_hackernews():
    [h] = parse_hits(
        {"hits": [{"objectID": "42", "title": "Show HN: X", "url": None, "points": 99, "num_comments": 5,
                   "created_at": "2026-01-02T03:04:05Z", "story_text": "<p>hello <b>world</b></p>"},
                  {"objectID": "43", "title": ""}]}
    )
    assert h["url"] == "https://news.ycombinator.com/item?id=42" and h["venue"] == "Hacker News ▲99"
    assert "99 points" in h["snippet"] and "hello world" in h["snippet"]


def test_parse_huggingface_papers_tolerates_shapes():
    data = [{"paper": {"id": "2401.00001", "title": "T  itle", "summary": "S", "upvotes": 12,
                       "publishedAt": "2024-01-01T00:00:00.000Z", "authors": [{"name": "A"}]}},
            {"title": "no id"}]
    [p] = parse_papers(data)
    assert p["key"] == "arxiv:2401.00001" and p["title"] == "T itle" and p["venue"].endswith("▲12")
    assert parse_papers({"papers": data})[0]["key"] == "arxiv:2401.00001"
    assert parse_papers({}) == []


def test_rank_key_prefers_relevance_then_citations():
    hi = {"relevance": 9}
    lo_cited = {"relevance": 7, "citation_count": 100000}
    assert _rank_key(hi) > _rank_key(lo_cited)  # 引用加成有上限，不会压过相关性
    assert _rank_key({"relevance": 7, "citation_count": 1000}) > _rank_key({"relevance": 7})
