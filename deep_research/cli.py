"""命令行入口。

示例：
    deep-research "多模态大模型的幻觉问题研究进展"
    deep-research "2026 年 AI Agent 框架业界格局" --review --iterations 2 --model gpt-4o
    deep-research --resume 3f9c1a2b      # 从断点继续上次中断的任务
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from . import settings
from .graph import build_graph
from .nodes import format_plan
from .reports import save_report

NODE_LABELS = {
    "plan": "📋 规划完成",
    "review_plan": "👀 计划已确认",
    "research": "🔍 子问题调研完成",
    "reflect": "🤔 反思完成",
    "write_report": "📝 报告撰写完成",
    "verify_report": "🔎 事实核验完成",
}


def _print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def _describe(node: str, update: dict) -> str:
    label = NODE_LABELS.get(node, node)
    if not isinstance(update, dict):
        return label
    if node == "plan":
        return f"{label}（类型: {update.get('research_type')}）\n{format_plan(update.get('plan', []))}"
    if node == "research":
        lines = []
        for f in update.get("findings", []):
            full = sum(1 for s in f["sources"] if s.get("fulltext"))
            extra = f"，{full} 篇读取了全文" if full else ""
            lines.append(
                f"{label} [{f['sub_question_id']}] {f['question']} —— {len(f['sources'])} 个来源{extra}"
            )
            if f.get("coverage"):
                lines.append(f"    ℹ️  {f['coverage']}")
            for e in f.get("errors", [])[:3]:
                lines.append(f"    ⚠️  {e[:160]}")
        return "\n".join(lines) or label
    if node == "reflect":
        tasks = update.get("pending_tasks") or []
        if tasks:
            return f"{label}：需要补充调研 {len(tasks)} 个子问题\n{format_plan(tasks)}"
        return f"{label}：材料充分，开始撰写报告"
    if node == "verify_report":
        v = update.get("verification") or {}
        if not v.get("checked"):
            return f"{label}：{v.get('error') or '已跳过'}"
        text = f"{label}：核验 {v['checked']} 条陈述 —— ✅ {v['supported']} · 🔶 {v['partial']} · ⚠️ {v['unsupported']}"
        return text + ("（已修订）" if v.get("revised") else "")
    return label


def _configurable(args: argparse.Namespace, thread_id: str) -> dict:
    cfg: dict = {"thread_id": thread_id, "human_review": args.review}
    mapping = {
        "llm_model": args.model,
        "writer_model": args.writer_model,
        "max_sub_questions": args.max_sub_questions,
        "max_iterations": args.iterations,
        "report_language": args.language,
        "verify_mode": args.verify,
        "enable_fulltext": False if args.no_fulltext else None,
        "enable_relevance_filter": False if args.no_filter else None,
    }
    cfg.update({k: v for k, v in mapping.items() if v is not None})
    return cfg


async def run(args: argparse.Namespace) -> int:
    data_dir = Path(os.getenv("DATA_DIR", ".deep_research"))
    data_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(data_dir / "checkpoints.db")) as saver:
        graph = build_graph(checkpointer=saver)
        resuming = bool(args.resume)
        thread_id = args.resume or uuid.uuid4().hex[:8]
        config = {"configurable": _configurable(args, thread_id), "recursion_limit": 100}

        if resuming:
            snap = await graph.aget_state(config)
            if not snap.values:
                _print(f"❌ 找不到任务 {thread_id} 的断点")
                return 1
            topic = snap.values.get("topic", "")
            payload: object = None
            _print(f"▶️  从断点继续：{topic}（任务 {thread_id}）\n")
        else:
            if not args.topic:
                _print("❌ 请提供调研课题，或使用 --resume 继续上次任务")
                return 2
            topic = args.topic
            payload = {"topic": topic}
            _print(f"🚀 开始调研：{topic}（任务 {thread_id}，中断后可用 --resume {thread_id} 继续）\n")

        while True:
            interrupted = None
            async for chunk in graph.astream(payload, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupted = update[0].value
                    else:
                        _print(_describe(node, update), "\n")
            if interrupted is None:
                break
            _print("=" * 60)
            _print(f"研究计划（类型: {interrupted.get('research_type')}）：")
            _print(interrupted.get("plan_text", ""))
            _print("=" * 60)
            answer = input(f"{interrupted.get('instruction', '')}\n> ")
            payload = Command(resume=answer)

        state = (await graph.aget_state(config)).values
    report: str = state.get("report", "")
    if not report:
        _print("❌ 未生成报告")
        return 1

    md_path = save_report(Path(args.output_dir), topic, state, report)
    if not args.quiet:
        _print("\n" + report)
    _print(f"\n✅ 报告已保存：{md_path.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    settings.apply()

    p = argparse.ArgumentParser(description="基于 LangGraph 的论文/业界进展深度调研 Agent")
    p.add_argument("topic", nargs="?", help="调研课题")
    p.add_argument("--resume", metavar="TASK_ID", help="从断点继续指定任务")
    p.add_argument("--review", action="store_true", help="规划后暂停，人工审核/修改研究计划")
    p.add_argument("--model", help="规划/检索所用模型（覆盖 LLM_MODEL）")
    p.add_argument("--writer-model", dest="writer_model", help="撰写报告所用模型")
    p.add_argument("--max-sub-questions", type=int, help="最多子问题数")
    p.add_argument("--iterations", type=int, help="反思补充检索最大轮数（0 表示不反思）")
    p.add_argument("--language", help="报告语言，默认中文")
    p.add_argument("--verify", choices=["off", "annotate", "revise"], help="事实核验模式（默认 annotate）")
    p.add_argument("--no-fulltext", action="store_true", help="不抓取全文，仅基于摘要")
    p.add_argument("--no-filter", action="store_true", help="关闭 LLM 相关性过滤")
    p.add_argument("-o", "--output-dir", default="reports", help="报告输出目录")
    p.add_argument("-q", "--quiet", action="store_true", help="不在终端打印完整报告")
    args = p.parse_args(argv)

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        _print("\n已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
