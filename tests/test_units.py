import pytest

from deep_research.citations import (
    build_global_sources,
    cited_numbers,
    finalize_report,
    remap_citations,
)
from deep_research.tools import dedupe_sources
from deep_research.tools.arxiv import build_query, parse_atom


def test_cited_numbers_variants():
    text = "A [1]. B [2, 3]. C【4】 D [5-7] E [1] link [x](http://a) F [8](http://b)"
    assert cited_numbers(text) == [1, 2, 3, 4, 5, 6, 7]


def test_remap_drops_unknown():
    assert remap_citations("x [1, 9] y [9] z [2]", {1: 5, 2: 3}) == "x [5] y  z [3]"


def test_global_sources_and_finalize():
    s = lambda k: {"key": k, "title": k.upper(), "url": f"http://{k}", "source_type": "web"}
    findings = [
        {"sub_question_id": "q1", "question": "", "summary": "a [1] b [2]", "sources": [s("a"), s("b")], "errors": []},
        {"sub_question_id": "q2", "question": "", "summary": "c [1] b [2]", "sources": [s("c"), s("b")], "errors": []},
    ]
    registry, summaries = build_global_sources(findings)
    assert [r["key"] for r in registry] == ["a", "b", "c"]
    assert summaries == ["a [1] b [2]", "c [3] b [2]"]

    report, refs = finalize_report("# T\nfoo [3] bar [1, 3] baz [42]", registry)
    assert [r["key"] for r in refs] == ["c", "a"]
    assert "foo [1] bar [1, 2] baz" in report
    assert "## 参考文献" in report and "[2] A." in report


def test_dedupe_merges_arxiv_and_s2():
    a = {"key": "arxiv:2401.00001", "source_type": "paper", "title": "Foo: A Bar", "snippet": "short"}
    b = {"key": "arxiv:2401.00001", "source_type": "paper", "title": "Foo: a bar", "snippet": "a much longer abstract", "citation_count": 12}
    c = {"key": "s2:xyz", "source_type": "paper", "title": "FOO a BAR!", "venue": "NeurIPS"}
    out = dedupe_sources([a, b, c])
    assert len(out) == 1
    assert out[0]["citation_count"] == 12 and out[0]["venue"] == "NeurIPS"
    assert out[0]["snippet"] == "a much longer abstract"


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <published>2024-01-22T00:00:00Z</published>
    <title>Deep   Research
      Agents</title>
    <summary> An abstract. </summary>
    <author><name>Alice</name></author><author><name>Bob</name></author>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_atom():
    [p] = parse_atom(ATOM)
    assert p["key"] == "arxiv:2401.12345"
    assert p["title"] == "Deep Research Agents"
    assert p["url"] == "https://arxiv.org/abs/2401.12345"
    assert p["authors"] == ["Alice", "Bob"] and p["venue"] == "arXiv (cs.CL)"


def test_build_query():
    assert build_query("retrieval augmented generation for LLM") == (
        "all:retrieval AND all:augmented AND all:generation AND all:LLM"
    )
