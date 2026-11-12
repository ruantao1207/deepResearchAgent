"""图状态与结构化输出 Schema。"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


class Source(TypedDict, total=False):
    """一条检索来源（论文或网页）。"""

    key: str  # 去重用的唯一键
    source_type: Literal["paper", "web"]
    origin: str  # arxiv / semantic_scholar / tavily / duckduckgo
    title: str
    url: str
    snippet: str  # 摘要 / 网页片段
    authors: list[str]
    published: str  # 发表日期或年份
    venue: str
    citation_count: int
    relevance: float  # LLM 相关性打分（0-10）
    fulltext: str  # 抓取到的全文/正文节选，用于总结与事实核验


class SubQuestion(TypedDict):
    id: str
    question: str
    rationale: str
    search_queries: list[str]
    sources: list[str]  # 使用的数据源：arxiv / semantic_scholar / web / github / huggingface / hackernews


class Finding(TypedDict):
    """单个子问题的调研结论。summary 中的 [n] 对应 sources[n-1]。"""

    sub_question_id: str
    question: str
    summary: str
    sources: list[Source]
    errors: list[str]
    coverage: str  # 覆盖度说明；空串表示正常


# ---------------------------------------------------------------------------
# 图状态
# ---------------------------------------------------------------------------


class InputState(TypedDict, total=False):
    topic: str
    local_docs: list[Source]


class OutputState(TypedDict):
    report: str
    references: list[Source]
    verification: dict


class ResearchState(TypedDict, total=False):
    topic: str
    local_docs: list[Source]  # 用户上传的本地文档候选源
    research_type: str  # academic / industry / mixed
    plan: list[SubQuestion]  # 所有子问题（含反思补充的）
    pending_tasks: list[SubQuestion]  # 待派发的子问题
    plan_feedback: str  # 人工审核意见；空字符串表示通过
    findings: Annotated[list[Finding], operator.add]  # 并行节点结果自动合并
    iteration: int
    report_body: str  # 不含参考文献的正文（引用号已是最终编号）
    report: str
    references: list[Source]
    verification: dict


class ResearchTaskState(TypedDict, total=False):
    """通过 Send 派发给单个研究节点的私有状态。"""

    topic: str
    task: SubQuestion
    local_docs: list[Source]


# ---------------------------------------------------------------------------
# LLM 结构化输出 Schema
# ---------------------------------------------------------------------------


class SubQuestionSpec(BaseModel):
    question: str = Field(description="子问题，具体、可检索")
    rationale: str = Field(description="为什么需要研究这个子问题")
    search_queries: list[str] = Field(
        description="2-3 个检索词。论文库检索用简短英文关键词（3-6 个词），业界检索可以是自然语句"
    )
    sources: list[Literal["arxiv", "semantic_scholar", "web", "github", "huggingface", "hackernews"]] = Field(
        description="该子问题适合的数据源"
    )


class ResearchPlan(BaseModel):
    research_type: Literal["academic", "industry", "mixed"] = Field(
        description="academic=偏论文；industry=偏产品/公司/开源动态；mixed=两者兼有"
    )
    sub_questions: list[SubQuestionSpec]


class Reflection(BaseModel):
    is_sufficient: bool = Field(description="现有材料是否足以写出高质量报告")
    knowledge_gaps: str = Field(description="缺失的关键信息，充分时可为空")
    follow_up_questions: list[SubQuestionSpec] = Field(
        default_factory=list, description="补充检索的子问题，充分时为空"
    )


class ItemScore(BaseModel):
    id: int = Field(description="候选资料编号")
    score: int = Field(ge=0, le=10, description="与子问题的相关性：0=无关，10=直接回答")


class RelevanceScores(BaseModel):
    scores: list[ItemScore]


class ClaimVerdict(BaseModel):
    id: int = Field(description="陈述编号")
    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="supported=来源明确支持；partial=只支持一部分或表述过度；unsupported=来源中找不到依据或与来源矛盾"
    )
    reason: str = Field(description="一句话理由，partial/unsupported 时说明问题所在")


class Verification(BaseModel):
    verdicts: list[ClaimVerdict]
