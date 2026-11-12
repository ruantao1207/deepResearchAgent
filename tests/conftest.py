"""共享测试夹具：假 LLM + 假检索 + 假全文抓取，整张图无需任何 API Key / 网络即可跑通。"""

import asyncio
import re

import pytest
from langchain_core.messages import AIMessage

from deep_research import llm as llm_mod
from deep_research import nodes, tools
from deep_research.state import (
    ClaimVerdict,
    ItemScore,
    Reflection,
    RelevanceScores,
    ResearchPlan,
    SubQuestionSpec,
    Verification,
)
from deep_research.web import server

WRITER_TEXT = "# 报告\n推测解码通过草稿模型加速推理 [2]。该方法在多个基准上取得显著加速 [1, 2]。乱引的陈述不应保留 [999]。"


class FakeStructured:
    def __init__(self, parent, schema):
        self.parent, self.schema = parent, schema

    async def ainvoke(self, messages):
        p, prompt = self.parent, messages[-1].content
        if self.schema is ResearchPlan:
            p.plan_calls.append(prompt)
            n = 3 if "修改意见" in prompt else 2
            return ResearchPlan(
                research_type="mixed",
                sub_questions=[
                    SubQuestionSpec(
                        question=f"子问题{i}",
                        rationale="r",
                        search_queries=[f"query {i}"],
                        sources=["arxiv", "web"],
                    )
                    for i in range(n)
                ],
            )
        if self.schema is Reflection:
            p.reflect_calls += 1
            if p.reflect_calls == 1:
                return Reflection(
                    is_sufficient=False,
                    knowledge_gaps="缺少业界",
                    follow_up_questions=[
                        SubQuestionSpec(question="补充问题", rationale="r", search_queries=["q"], sources=["web"])
                    ],
                )
            return Reflection(is_sufficient=True, knowledge_gaps="", follow_up_questions=[])
        if self.schema is RelevanceScores:
            p.relevance_calls += 1
            n = len(re.findall(r"^\[\d+\]", prompt, flags=re.M))
            return RelevanceScores(scores=[ItemScore(id=i, score=8 if i < n else 1) for i in range(1, n + 1)])
        if self.schema is Verification:
            p.verify_calls += 1
            ids = [int(x) for x in re.findall(r"^(\d+)\. ", prompt, flags=re.M)]
            return Verification(
                verdicts=[
                    ClaimVerdict(id=i, verdict="unsupported" if "显著加速" in _claim(prompt, i) else "supported", reason="来源未提及“显著”")
                    for i in ids
                ]
            )
        raise AssertionError(self.schema)


def _claim(prompt: str, i: int) -> str:
    m = re.search(rf"^{i}\. (.*)$", prompt, flags=re.M)
    return m.group(1) if m else ""


class FakeLLM:
    def __init__(self):
        self.plan_calls = []
        self.reflect_calls = 0
        self.relevance_calls = 0
        self.verify_calls = 0
        self.writer_prompts = []
        self.research_prompts = []
        self.revise_calls = 0
        self.fail_research = 0  # 前 N 次研究笔记调用抛错
        self.research_delay = 0.0

    def with_structured_output(self, schema, method=None):
        return FakeStructured(self, schema)

    async def ainvoke(self, messages):
        prompt = messages[-1].content if isinstance(messages, list) else str(messages)
        if "请修订报告" in prompt:
            self.revise_calls += 1
            return AIMessage(prompt.split("## 报告正文\n", 1)[1].replace("显著加速", "一定加速"))
        if "可引用来源列表" in prompt:
            self.writer_prompts.append(prompt)
            return AIMessage(WRITER_TEXT)
        if "## 候选资料" in prompt:
            self.research_prompts.append(prompt)
            if self.research_delay:
                await asyncio.sleep(self.research_delay)
            if self.fail_research > 0:
                self.fail_research -= 1
                raise RuntimeError("simulated llm failure")
            return AIMessage("要点 A [2]；要点 B [1, 2]；无效 [50]。")
        return AIMessage("ok")


async def fake_arxiv(query, limit):
    return [
        {"key": f"arxiv:{query}", "source_type": "paper", "origin": "arxiv", "title": f"Paper {query}",
         "url": f"https://arxiv.org/abs/{query}", "snippet": "abs", "authors": ["A"], "published": "2025-01-01"}
    ]


async def fake_web(query, limit):
    return [
        {"key": f"url:{query}", "source_type": "web", "origin": "tavily", "title": f"News {query}",
         "url": f"https://news/{query}", "snippet": "body"}
    ]


async def failing(query, limit):
    raise RuntimeError("boom")


async def fake_fulltext(src, max_chars=3500, timeout=25.0):
    return f"全文节选：{src.get('title', '')} " * 30


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(llm_mod, "get_llm", lambda *a, **k: f)
    monkeypatch.setitem(tools.SEARCHERS, "arxiv", fake_arxiv)
    monkeypatch.setitem(tools.SEARCHERS, "web", fake_web)
    monkeypatch.setitem(tools.SEARCHERS, "semantic_scholar", failing)
    monkeypatch.setattr(nodes, "fetch_fulltext", fake_fulltext)
    return f


@pytest.fixture(autouse=True)
def isolate_server(tmp_path, monkeypatch):
    """Web 服务的数据目录 / 报告目录指向临时目录，避免污染工作区。"""
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(server, "REPORTS_DIR", tmp_path / "reports")
