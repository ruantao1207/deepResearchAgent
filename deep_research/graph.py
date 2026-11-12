"""组装 LangGraph 状态图。

    START
      │
      ▼
    plan ◄──────────────┐ (有修改意见则重新规划)
      │                 │
      ▼                 │
    review_plan ────────┘
      │  Send × N（每个子问题并行）
      ▼
    research ◄──────────┐
      │                 │ Send × M（补充子问题）
      ▼                 │
    reflect ────────────┘
      │ (材料充分 / 达到最大轮数)
      ▼
    write_report ──► verify_report ──► END
      (逐条核对带引用的陈述；annotate 标注 / revise 修订)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from .config import Configuration
from .nodes import (
    plan_research,
    reflect,
    research_sub_question,
    review_plan,
    route_after_reflect,
    route_after_review,
    verify_report,
    write_report,
)
from .state import InputState, OutputState, ResearchState


def build_graph(checkpointer=None):
    builder = StateGraph(
        ResearchState,
        input_schema=InputState,
        output_schema=OutputState,
    )
    llm_retry = RetryPolicy(max_attempts=2)

    builder.add_node("plan", plan_research, retry_policy=llm_retry)
    builder.add_node("review_plan", review_plan)
    builder.add_node("research", research_sub_question, retry_policy=llm_retry)
    builder.add_node("reflect", reflect, retry_policy=llm_retry)
    builder.add_node("write_report", write_report, retry_policy=llm_retry)
    builder.add_node("verify_report", verify_report)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "review_plan")
    builder.add_conditional_edges("review_plan", route_after_review, ["plan", "research"])
    builder.add_edge("research", "reflect")
    builder.add_conditional_edges("reflect", route_after_reflect, ["research", "write_report"])
    builder.add_edge("write_report", "verify_report")
    builder.add_edge("verify_report", END)

    return builder.compile(checkpointer=checkpointer, name="deep_research")


# 供 `langgraph dev` / LangGraph Platform 使用（平台自动注入持久化）
graph = build_graph()

__all__ = ["build_graph", "graph", "Configuration"]
