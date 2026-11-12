"""运行时配置。

优先级：RunnableConfig["configurable"] > 环境变量 > 默认值。
这样既可以通过 .env 全局配置，也可以在每次调用 graph 时按需覆盖。
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

load_dotenv()

SourceName = Literal["arxiv", "semantic_scholar", "web", "github", "huggingface", "hackernews"]

# 字段名 -> 环境变量名
_ENV_MAP = {
    "llm_model": "LLM_MODEL",
    "writer_model": "WRITER_MODEL",
    "temperature": "LLM_TEMPERATURE",
    "max_sub_questions": "MAX_SUB_QUESTIONS",
    "max_iterations": "MAX_ITERATIONS",
    "results_per_query": "RESULTS_PER_QUERY",
    "max_sources_per_task": "MAX_SOURCES_PER_TASK",
    "max_candidates": "MAX_CANDIDATES",
    "report_language": "REPORT_LANGUAGE",
    "human_review": "HUMAN_REVIEW",
    "structured_output_method": "STRUCTURED_OUTPUT_METHOD",
    "enable_relevance_filter": "ENABLE_RELEVANCE_FILTER",
    "min_relevance": "MIN_RELEVANCE",
    "enable_fulltext": "ENABLE_FULLTEXT",
    "max_fulltext_sources": "MAX_FULLTEXT_SOURCES",
    "fulltext_chars": "FULLTEXT_CHARS",
    "verify_mode": "VERIFY_MODE",
    "max_verify_claims": "MAX_VERIFY_CLAIMS",
}

_BOOL_FIELDS = {"human_review", "enable_relevance_filter", "enable_fulltext"}


class Configuration(BaseModel):
    """Agent 可调参数。"""

    llm_model: str = Field("gpt-4o-mini", description="规划/检索/反思所用模型")
    writer_model: str | None = Field(None, description="撰写报告所用模型，默认同 llm_model")
    temperature: float | None = Field(None, description="采样温度，None 表示不传")

    max_sub_questions: int = Field(5, ge=1, le=12, description="规划阶段最多拆分的子问题数")
    max_iterations: int = Field(1, ge=0, le=5, description="反思-补充检索的最大轮数")
    results_per_query: int = Field(5, ge=1, le=20, description="每个查询每个数据源返回条数")
    max_sources_per_task: int = Field(8, ge=1, le=20, description="每个子问题最终保留的来源数")
    max_candidates: int = Field(18, ge=4, le=50, description="每个子问题交给 LLM 筛选的候选数")

    report_language: str = Field("中文", description="报告语言")
    human_review: bool = Field(False, description="是否在规划后暂停等待人工审核")
    structured_output_method: Literal["function_calling", "json_mode", "json_schema"] = (
        "function_calling"
    )

    # --- 检索质量 ---
    enable_relevance_filter: bool = Field(True, description="用 LLM 对候选资料做相关性打分并过滤")
    min_relevance: int = Field(5, ge=0, le=10, description="相关性分数低于该值的候选被丢弃（至少保留 3 条）")
    enable_fulltext: bool = Field(True, description="抓取高相关来源的全文/网页正文用于总结")
    max_fulltext_sources: int = Field(3, ge=0, le=10, description="每个子问题最多抓取全文的来源数")
    fulltext_chars: int = Field(3500, ge=500, le=20000, description="每个来源喂给 LLM 的全文节选字数")

    # --- 事实核验 ---
    verify_mode: Literal["off", "annotate", "revise"] = Field(
        "annotate", description="off=不核验；annotate=标注存疑陈述；revise=核验后让写作模型修订"
    )
    max_verify_claims: int = Field(36, ge=4, le=120, description="最多核验的带引用陈述数")

    @property
    def effective_writer_model(self) -> str:
        return self.writer_model or self.llm_model

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        configurable: dict[str, Any] = (config or {}).get("configurable", {}) or {}
        values: dict[str, Any] = {}
        for name in cls.model_fields:
            if configurable.get(name) is not None:
                values[name] = configurable[name]
                continue
            env_val = os.getenv(_ENV_MAP.get(name, name.upper()))
            if env_val not in (None, ""):
                values[name] = _coerce_bool(env_val) if name in _BOOL_FIELDS else env_val
        return cls(**values)


def _coerce_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}
