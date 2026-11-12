import asyncio

import httpx
import pytest

from deep_research.tools import openalex, semantic_scholar as s2


def test_openalex_parse_and_abstract():
    data = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.48550/arXiv.2401.00001",
                "title": "Gen Rec",
                "abstract_inverted_index": {"world": [1], "Hello": [0]},
                "publication_date": "2024-01-02",
                "primary_location": {"source": {"display_name": "NeurIPS"}},
                "authorships": [{"author": {"display_name": "Alice"}}],
                "cited_by_count": 7,
            },
            {"id": "https://openalex.org/W2", "doi": "https://doi.org/10.1/x", "title": "Other"},
            {"id": "https://openalex.org/W3", "title": ""},
        ]
    }
    a, b = openalex.parse_works(data)
    assert a["key"] == "arxiv:2401.00001" and a["url"] == "https://arxiv.org/abs/2401.00001"
    assert a["snippet"] == "Hello world" and a["citation_count"] == 7 and a["venue"] == "NeurIPS"
    assert b["key"] == "doi:10.1/x"


class FakeResp:
    def __init__(self, status, headers=None, payload=None):
        self.status_code, self.headers, self._p = status, headers or {}, payload or {"data": []}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("GET", "http://x"), response=httpx.Response(self.status_code))

    def json(self):
        return self._p


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(s2, "_blocked_until", 0.0)
    monkeypatch.setattr(s2, "MIN_INTERVAL", 0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(s2.asyncio, "sleep", lambda *_: real_sleep(0))


def patch_client(monkeypatch, responses, counter):
    class C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, *a, **k):
            counter.append(1)
            return responses.pop(0) if len(responses) > 1 else responses[0]

    monkeypatch.setattr(s2.httpx, "AsyncClient", C)


def test_429_falls_back_and_trips_breaker(monkeypatch):
    calls, fallback = [], []

    async def fake_openalex(q, n):
        fallback.append(q)
        return [{"key": "doi:1", "title": "from openalex"}]

    monkeypatch.setattr(s2, "search_openalex", fake_openalex)
    patch_client(monkeypatch, [FakeResp(429)], calls)

    r = asyncio.run(s2.search_semantic_scholar("q1", 3))
    assert r[0]["title"] == "from openalex" and len(calls) == s2.MAX_RETRIES + 1

    # 熔断期内：不再请求 S2，直接回退
    r = asyncio.run(s2.search_semantic_scholar("q2", 3))
    assert len(calls) == s2.MAX_RETRIES + 1 and fallback == ["q1", "q2"]


def test_success_after_retry(monkeypatch):
    calls = []
    payload = {"data": [{"title": "T", "paperId": "p1", "externalIds": {}}]}
    patch_client(monkeypatch, [FakeResp(429, {"Retry-After": "1"}), FakeResp(200, payload=payload)], calls)
    r = asyncio.run(s2.search_semantic_scholar("q", 3))
    assert r[0]["origin"] == "semantic_scholar" and len(calls) == 2
