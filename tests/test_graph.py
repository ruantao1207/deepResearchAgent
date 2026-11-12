"""端到端测试：整张图（含并行派发、相关性过滤、全文、反思、核验、人工审核）。"""

import asyncio

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_research.graph import build_graph


def run(graph, payload, config):
    return asyncio.run(graph.ainvoke(payload, config))


def test_full_run_with_reflection_filter_fulltext_and_verify(fake):
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1", "max_iterations": 2}}
    out = run(graph, {"topic": "测试课题"}, config)

    state = graph.get_state(config).values
    findings = sorted(state["findings"], key=lambda f: f["sub_question_id"])
    assert [f["sub_question_id"] for f in findings] == ["q1", "q2", "q3"]
    assert fake.reflect_calls == 2

    for f in findings:
        assert "[50]" not in f["summary"]
        assert len(f["sources"]) == 2
        assert "放宽检索" in f["coverage"]  # 初次结果过少 -> 自动放宽
        for s in f["sources"]:
            assert s["fulltext"].startswith("全文节选")  # 全文已抓取并保留
    # q1/q2 放宽后有 4 个候选 -> 触发相关性过滤；q3 只有 3 个候选 -> 不过滤
    assert fake.relevance_calls == 2
    assert all("relevance" in s for f in findings[:2] for s in f["sources"])
    # 研究 prompt 中带有全文节选
    assert any("全文节选" in p for p in fake.research_prompts)
    # 写作 prompt 带有覆盖度提示
    assert "覆盖度提示" in fake.writer_prompts[0]

    report = out["report"]
    assert "[999]" not in report
    assert len(out["references"]) == 2
    # 核验：含"显著加速"的陈述被判 unsupported，并被标注 + 附录说明
    v = out["verification"]
    assert v["checked"] == 2 and v["supported"] == 1 and v["unsupported"] == 1
    assert "显著加速 [1, 2]。 ⚠️" in report
    assert "## 事实核验说明" in report
    assert report.index("## 事实核验说明") < report.index("## 参考文献")
    assert report.index("推测解码通过草稿模型加速推理 [1]") < report.index("## 参考文献")


def test_verify_revise_mode(fake):
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t2", "max_iterations": 0, "verify_mode": "revise"}}
    out = run(graph, {"topic": "x"}, config)
    assert fake.revise_calls == 1
    body, appendix = out["report"].split("## 事实核验说明")
    assert "一定加速" in body and "显著加速" not in body  # 正文已修订
    assert "⚠️" not in body  # 修订成功时不再保留标注
    assert "显著加速" in appendix and "已修订" in appendix  # 附录保留原文供读者对照
    assert out["verification"]["revised"] is True


def test_features_can_be_disabled(fake):
    graph = build_graph(checkpointer=InMemorySaver())
    config = {
        "configurable": {
            "thread_id": "t3",
            "max_iterations": 0,
            "verify_mode": "off",
            "enable_fulltext": False,
            "enable_relevance_filter": False,
        }
    }
    out = run(graph, {"topic": "x"}, config)
    assert fake.reflect_calls == 0 and fake.relevance_calls == 0 and fake.verify_calls == 0
    assert out["verification"]["checked"] == 0
    assert "事实核验" not in out["report"]
    state = graph.get_state(config).values
    assert all(not s.get("fulltext") for f in state["findings"] for s in f["sources"])


def test_human_review_replan_then_approve(fake):
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t4", "human_review": True, "max_iterations": 0}}

    async def go():
        await graph.ainvoke({"topic": "x"}, config)
        s = await graph.aget_state(config)
        assert s.interrupts and s.interrupts[0].value["type"] == "plan_review"
        await graph.ainvoke(Command(resume="请增加一个关于评测的子问题"), config)
        s = await graph.aget_state(config)
        assert s.interrupts and len(s.values["plan"]) == 3
        return await graph.ainvoke(Command(resume="通过"), config)

    out = asyncio.run(go())
    assert len(fake.plan_calls) == 2 and "修改意见" in fake.plan_calls[1]
    assert len(graph.get_state(config).values["findings"]) == 3
    assert out["report"].startswith("# 报告")


def test_relevance_filter_failure_is_not_fatal(fake, monkeypatch):
    original = fake.with_structured_output

    def broken(schema, method=None):
        from deep_research.state import RelevanceScores

        if schema is RelevanceScores:
            class Boom:
                async def ainvoke(self, _):
                    raise ValueError("bad json")
            return Boom()
        return original(schema, method)

    monkeypatch.setattr(fake, "with_structured_output", broken)
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t5", "max_iterations": 0}}
    out = run(graph, {"topic": "x"}, config)
    assert out["report"].startswith("# 报告")
    errs = [e for f in graph.get_state(config).values["findings"] for e in f["errors"]]
    assert any("relevance_filter" in e for e in errs)
