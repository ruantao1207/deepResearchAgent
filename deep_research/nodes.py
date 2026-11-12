"""图节点实现。

流程：
    plan -> review_plan -(Send x N)-> research -> reflect -(Send x M | 结束)-> write_report -> verify_report
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import date

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send, interrupt

from . import llm as llm_mod
from . import tools
from . import verify as vf
from .citations import (
    append_references,
    build_global_sources,
    cited_numbers,
    format_source_brief,
    remap_citations,
    renumber_report,
)
from .config import Configuration
from .prompts import (
    PLAN_FEEDBACK_SECTION,
    PLANNER_PROMPT,
    REFLECTION_PROMPT,
    RELEVANCE_PROMPT,
    RESEARCHER_PROMPT,
    REVISE_PROMPT,
    VERIFY_PROMPT,
    WRITER_PROMPT,
)
from .state import (
    Finding,
    Reflection,
    RelevanceScores,
    ResearchPlan,
    ResearchState,
    ResearchTaskState,
    Source,
    SubQuestion,
    SubQuestionSpec,
    Verification,
)
from .tools.fulltext import fetch_fulltext

logger = logging.getLogger(__name__)

APPROVE_WORDS = {"", "y", "yes", "ok", "approve", "approved", "通过", "同意", "确认", "可以", "好"}
MIN_CANDIDATES = 4  # 去重后候选少于该值时触发放宽检索
MIN_KEEP = 3  # 相关性过滤后至少保留的条数
VERIFY_BATCH = 10
STREAM_CALLBACK: ContextVar[Callable[[str], Awaitable[None]] | None] = ContextVar(
    "stream_callback", default=None
)


def _today() -> str:
    return date.today().isoformat()


def _text(msg) -> str:
    """兼容 content 为字符串或内容块列表的情况。"""
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _to_sub_questions(specs: list[SubQuestionSpec], start: int) -> list[SubQuestion]:
    out: list[SubQuestion] = []
    for i, s in enumerate(specs, start=start):
        srcs = [x for x in dict.fromkeys(s.sources) if x in tools.SEARCHERS] or ["web"]
        queries = [q.strip() for q in s.search_queries if q.strip()][:3] or [s.question]
        out.append(
            SubQuestion(
                id=f"q{i}",
                question=s.question,
                rationale=s.rationale,
                search_queries=queries,
                sources=srcs,
            )
        )
    return out


def format_plan(plan: list[SubQuestion]) -> str:
    lines = []
    for q in plan:
        lines.append(f"- [{q['id']}] {q['question']}")
        lines.append(f"    数据源: {', '.join(q['sources'])} | 检索词: {'; '.join(q['search_queries'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. 规划
# ---------------------------------------------------------------------------


async def plan_research(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    feedback = (state.get("plan_feedback") or "").strip()
    feedback_section = (
        PLAN_FEEDBACK_SECTION.format(
            previous_plan=format_plan(state.get("plan") or []), feedback=feedback
        )
        if feedback
        else ""
    )
    prompt = PLANNER_PROMPT.format(
        today=_today(),
        topic=state["topic"],
        max_sub_questions=cfg.max_sub_questions,
        feedback_section=feedback_section,
    )
    llm = llm_mod.get_llm(cfg.llm_model, cfg.temperature)
    result: ResearchPlan = await llm_mod.ainvoke_structured(
        llm, ResearchPlan, prompt, cfg.structured_output_method, cfg.llm_model
    )
    plan = _to_sub_questions(result.sub_questions[: cfg.max_sub_questions], start=1)
    return {
        "research_type": result.research_type,
        "plan": plan,
        "pending_tasks": plan,
        "plan_feedback": "",
        "iteration": 0,
    }


# ---------------------------------------------------------------------------
# 2. 人工审核（可选，Human-in-the-loop）
# ---------------------------------------------------------------------------


async def review_plan(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    if not cfg.human_review:
        return {"plan_feedback": ""}
    answer = interrupt(
        {
            "type": "plan_review",
            "research_type": state.get("research_type"),
            "plan": state.get("plan", []),
            "plan_text": format_plan(state.get("plan", [])),
            "instruction": "回车/输入'通过'确认计划；或输入修改意见以重新规划。",
        }
    )
    text = str(answer or "").strip()
    return {"plan_feedback": "" if text.lower() in APPROVE_WORDS else text}


def route_after_review(state: ResearchState):
    if state.get("plan_feedback"):
        return "plan"
    return _dispatch(state)


def _dispatch(state: ResearchState) -> list[Send]:
    local_docs = state.get("local_docs") or []
    return [
        Send("research", {"topic": state["topic"], "task": task, "local_docs": local_docs})
        for task in state.get("pending_tasks", [])
    ]


# ---------------------------------------------------------------------------
# 3. 子问题研究（并行执行）：检索 -> 放宽重试 -> 相关性过滤 -> 全文抓取 -> 总结
# ---------------------------------------------------------------------------


def _rank_key(src: Source) -> float:
    """排序分：LLM 相关性为主，引用数 / 新近度做小幅加成。"""
    score = float(src.get("relevance", 5))
    score += min(1.5, math.log10(1 + (src.get("citation_count") or 0)) * 0.5)
    year = (src.get("published") or "")[:4]
    if year.isdigit() and int(year) >= date.today().year - 1:
        score += 0.5
    return score


async def _filter_by_relevance(
    topic: str, question: str, candidates: list[Source], cfg: Configuration, errors: list[str]
) -> list[Source]:
    brief = "\n".join(format_source_brief(i, s, snippet_chars=350) for i, s in enumerate(candidates, 1))
    prompt = RELEVANCE_PROMPT.format(topic=topic, question=question, candidates=brief)
    try:
        llm = llm_mod.get_llm(cfg.llm_model, cfg.temperature)
        result: RelevanceScores = await llm_mod.ainvoke_structured(
            llm, RelevanceScores, prompt, cfg.structured_output_method, cfg.llm_model
        )
    except Exception as e:  # noqa: BLE001  过滤失败不应中断研究
        errors.append(f"relevance_filter -> {type(e).__name__}: {str(e)[:150]}")
        return candidates

    scores = {s.id: s.score for s in result.scores if 1 <= s.id <= len(candidates)}
    if not scores:
        return candidates
    scored: list[Source] = []
    for i, src in enumerate(candidates, start=1):
        item = dict(src)
        if src.get("origin") == "local":
            item["relevance"] = 10.0
        else:
            item["relevance"] = float(scores.get(i, 3))  # 模型漏评的按"沾边"处理
        scored.append(item)  # type: ignore[arg-type]
    scored.sort(key=_rank_key, reverse=True)
    keep = [s for s in scored if s["relevance"] >= cfg.min_relevance]
    if len(keep) < MIN_KEEP:
        keep = scored[:MIN_KEEP]
    return keep[: cfg.max_sources_per_task + 4]


async def _attach_fulltext(candidates: list[Source], cfg: Configuration) -> list[Source]:
    k = min(cfg.max_fulltext_sources, len(candidates))
    if k <= 0:
        return candidates
    texts = await asyncio.gather(
        *(
            asyncio.sleep(0, result=c.get("fulltext"))
            if c.get("fulltext")
            else fetch_fulltext(c, cfg.fulltext_chars)
            for c in candidates[:k]
        ),
        return_exceptions=True,
    )
    out = list(candidates)
    for i, t in enumerate(texts):
        if isinstance(t, str) and t:
            item = dict(out[i])
            item["fulltext"] = t
            out[i] = item  # type: ignore[assignment]
    return out


async def research_sub_question(state: ResearchTaskState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    task = state["task"]
    notes: list[str] = []  # 覆盖度说明

    raw, errors = await tools.search_all(
        task["search_queries"], task["sources"], cfg.results_per_query
    )
    candidates: list[Source] = tools.dedupe_sources(raw)

    # 结果太少：放宽检索（直接用子问题原文，并补充 web + arxiv）
    if len(candidates) < MIN_CANDIDATES:
        wide_sources = list(dict.fromkeys([*task["sources"], "web", "arxiv"]))
        extra, extra_err = await tools.search_all(
            [task["question"][:200]], wide_sources, cfg.results_per_query
        )
        errors += extra_err
        candidates = tools.dedupe_sources([*candidates, *extra])
        notes.append("初次检索结果过少，已自动放宽检索词与数据源")
    candidates = candidates[: cfg.max_candidates]
    local_docs: list[Source] = [
        {**d, "relevance": 10.0} for d in (state.get("local_docs") or []) if isinstance(d, dict)
    ]
    if local_docs:
        candidates = tools.dedupe_sources([*local_docs, *candidates])

    if not candidates:
        return {
            "findings": [
                Finding(
                    sub_question_id=task["id"],
                    question=task["question"],
                    summary="（未检索到相关资料）",
                    sources=[],
                    errors=errors,
                    coverage="；".join([*notes, "未检索到任何资料，本部分结论缺失"]),
                )
            ]
        }

    if cfg.enable_relevance_filter and len(candidates) > MIN_KEEP:
        before = len(candidates)
        candidates = await _filter_by_relevance(state["topic"], task["question"], candidates, cfg, errors)
        logger.info("[%s] relevance filter: %d -> %d", task["id"], before, len(candidates))

    if cfg.enable_fulltext:
        candidates = await _attach_fulltext(candidates, cfg)

    candidates_text = "\n".join(
        format_source_brief(i, s, snippet_chars=700, fulltext_chars=cfg.fulltext_chars)
        for i, s in enumerate(candidates, start=1)
    )
    coverage_note = (
        "\n- 注意：本次检索覆盖有限（" + "；".join(notes) + "），请在结论中明确标注不确定性。" if notes else ""
    )
    prompt = RESEARCHER_PROMPT.format(
        today=_today(),
        topic=state["topic"],
        question=task["question"],
        candidates=candidates_text,
        language=cfg.report_language,
        coverage_note=coverage_note,
    )
    resp = await llm_mod.get_llm(cfg.llm_model, cfg.temperature).ainvoke([HumanMessage(prompt)])
    summary = _text(resp).strip()

    # 只保留被引用的来源，并重新编号为 1..k
    used = [n for n in cited_numbers(summary) if 1 <= n <= len(candidates)]
    used = used[: cfg.max_sources_per_task]
    mapping = {old: new for new, old in enumerate(used, start=1)}
    kept = [candidates[old - 1] for old in used]
    summary = remap_citations(summary, mapping)
    if len(kept) < 2:
        notes.append(f"可引用的有效来源仅 {len(kept)} 条")

    return {
        "findings": [
            Finding(
                sub_question_id=task["id"],
                question=task["question"],
                summary=summary,
                sources=kept,
                errors=errors,
                coverage="；".join(notes),
            )
        ]
    }


# ---------------------------------------------------------------------------
# 4. 反思：识别知识缺口，必要时补充检索
# ---------------------------------------------------------------------------


def _findings_text(state: ResearchState) -> tuple[list[Source], str]:
    registry, summaries = build_global_sources(state.get("findings", []))
    blocks = [
        f"### [{f['sub_question_id']}] {f['question']}\n{s}"
        for f, s in zip(state.get("findings", []), summaries)
    ]
    return registry, "\n\n".join(blocks)


async def reflect(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    iteration = state.get("iteration", 0)
    if iteration >= cfg.max_iterations:
        return {"pending_tasks": []}

    _, findings_text = _findings_text(state)
    max_follow_ups = max(1, cfg.max_sub_questions // 2)
    prompt = REFLECTION_PROMPT.format(
        today=_today(),
        topic=state["topic"],
        findings=findings_text,
        max_follow_ups=max_follow_ups,
    )
    llm = llm_mod.get_llm(cfg.llm_model, cfg.temperature)
    result: Reflection = await llm_mod.ainvoke_structured(
        llm, Reflection, prompt, cfg.structured_output_method, cfg.llm_model
    )

    if result.is_sufficient or not result.follow_up_questions:
        return {"pending_tasks": [], "iteration": iteration + 1}

    plan = state.get("plan", [])
    new_tasks = _to_sub_questions(result.follow_up_questions[:max_follow_ups], start=len(plan) + 1)
    return {"plan": plan + new_tasks, "pending_tasks": new_tasks, "iteration": iteration + 1}


def route_after_reflect(state: ResearchState):
    if state.get("pending_tasks"):
        return _dispatch(state)
    return "write_report"


# ---------------------------------------------------------------------------
# 5. 撰写报告
# ---------------------------------------------------------------------------


async def write_report(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    registry, findings_text = _findings_text(state)
    sources_text = "\n".join(
        format_source_brief(i, s) for i, s in enumerate(registry, start=1)
    )
    gaps = [
        f"- 子问题「{f['question']}」：{f['coverage']}"
        for f in state.get("findings", [])
        if f.get("coverage")
    ]
    coverage_section = (
        "\n## 覆盖度提示（请在报告相应部分明确说明资料有限，不要强行下结论）\n" + "\n".join(gaps) + "\n"
        if gaps
        else ""
    )
    prompt = WRITER_PROMPT.format(
        today=_today(),
        topic=state["topic"],
        research_type=state.get("research_type", "mixed"),
        findings=findings_text or "（无）",
        sources=sources_text or "（无）",
        coverage_section=coverage_section,
        language=cfg.report_language,
    )
    writer = llm_mod.get_llm(cfg.effective_writer_model, cfg.temperature)
    cb = STREAM_CALLBACK.get()
    raw_report = ""
    if cb is not None and hasattr(writer, "astream") and type(writer).__name__ != "FakeLLM":
        try:
            chunks: list[str] = []
            last_emit = 0.0
            async for chunk in writer.astream([HumanMessage(prompt)]):
                piece = _text(chunk)
                if piece:
                    chunks.append(piece)
                    now = time.monotonic()
                    if now - last_emit >= 0.25:
                        last_emit = now
                        await cb("".join(chunks))
            raw_report = "".join(chunks).strip()
            if raw_report:
                await cb(raw_report)
        except Exception as e:  # noqa: BLE001
            logger.warning("write_report stream fallback to ainvoke: %s", e)
            raw_report = ""
    if not raw_report:
        resp = await writer.ainvoke([HumanMessage(prompt)])
        raw_report = _text(resp).strip()
    body, refs = renumber_report(raw_report, registry)
    return {
        "report_body": body,
        "report": append_references(body, refs),
        "references": refs,
        "verification": {},
    }


# ---------------------------------------------------------------------------
# 6. 事实核验：逐条核对"带引用的陈述"与来源内容
# ---------------------------------------------------------------------------


async def _verify_batch(
    llm, cfg: Configuration, claims: list[vf.Claim], refs: list[Source], errors: list[str]
) -> None:
    numbers = sorted({n for c in claims for n in c.cites})
    prompt = VERIFY_PROMPT.format(
        claims="\n".join(f"{c.id}. {c.text}" for c in claims),
        evidence=vf.evidence_block(refs, numbers),
    )
    try:
        result: Verification = await llm_mod.ainvoke_structured(
            llm, Verification, prompt, cfg.structured_output_method, cfg.llm_model
        )
    except Exception as e:  # noqa: BLE001
        errors.append(f"{type(e).__name__}: {str(e)[:150]}")
        return
    by_id = {c.id: c for c in claims}
    for v in result.verdicts:
        c = by_id.get(v.id)
        if c is not None:
            c.verdict, c.reason = v.verdict, v.reason.strip()


async def verify_report(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    body: str = state.get("report_body") or ""
    refs: list[Source] = state.get("references") or []
    if cfg.verify_mode == "off" or not body or not refs:
        return {"verification": {"mode": cfg.verify_mode, "checked": 0}}

    all_claims = vf.extract_claims(body)
    claims, skipped = vf.sample_claims(all_claims, cfg.max_verify_claims)
    if not claims:
        return {"verification": {"mode": cfg.verify_mode, "checked": 0}}

    llm = llm_mod.get_llm(cfg.llm_model, cfg.temperature)
    errors: list[str] = []
    await asyncio.gather(
        *(_verify_batch(llm, cfg, b, refs, errors) for b in vf.batch(claims, VERIFY_BATCH))
    )
    stats = vf.tally(claims, skipped)
    if stats.checked == 0:  # 核验调用全部失败：保持原报告
        return {
            "verification": {"mode": cfg.verify_mode, "checked": 0, "errors": errors, "error": "核验未能完成"}
        }

    issues = [c for c in claims if c.verdict in ("unsupported", "partial")]
    revised_applied = False
    new_body = body
    if cfg.verify_mode == "revise" and issues:
        issue_text = "\n".join(f"- ({c.verdict}) {c.text}\n  问题：{c.reason}" for c in issues)
        revised = None
        try:
            writer = llm_mod.get_llm(cfg.effective_writer_model, cfg.temperature)
            resp = await writer.ainvoke(
                [HumanMessage(REVISE_PROMPT.format(issues=issue_text, report=body))]
            )
            revised = vf.sanitize_revision(_text(resp), body, len(refs))
        except Exception as e:  # noqa: BLE001
            errors.append(f"revise -> {type(e).__name__}: {str(e)[:150]}")
        if revised:
            new_body, revised_applied = revised, True
            for c in issues:
                c.fixed = True
        else:
            errors.append("修订稿未通过校验，已退回为标注模式")
    if not revised_applied:
        new_body = vf.annotate_body(body, claims)

    appendix = vf.build_appendix(claims, stats, "revise" if revised_applied else "annotate")
    return {
        "report_body": new_body,
        "report": append_references(new_body, refs, appendix),
        "verification": {
            "mode": cfg.verify_mode,
            "checked": stats.checked,
            "supported": stats.supported,
            "partial": stats.partial,
            "unsupported": stats.unsupported,
            "skipped": stats.skipped,
            "revised": revised_applied,
            "errors": errors,
            "issues": [
                {"claim": c.text, "verdict": c.verdict, "reason": c.reason, "fixed": c.fixed}
                for c in issues
            ],
        },
    }
