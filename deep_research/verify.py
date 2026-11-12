"""事实核验的纯函数部分：抽取带引用的陈述、抽样、标注、生成核验附录。

LLM 调用在 nodes.verify_report 中完成，这里不依赖网络，便于单元测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .citations import cited_numbers, remap_citations

_CITE = r"[\[【]\d+(?:\s*[,，、;；\-–]\s*\d+)*[\]】]"
# 一个句子 = 到句末标点为止，且句末标点后紧跟的引用标记（如"。[2]"）归属该句
_SENT = re.compile(rf".+?(?:(?:[。！？；!?;]|[.](?=\s|$))(?:\s*{_CITE})*|$)")
_MIN_LEN = 12

ICON = {"supported": "✅", "partial": "🔶", "unsupported": "⚠️"}


@dataclass
class Claim:
    id: int
    text: str
    cites: list[int]
    verdict: str = ""  # supported / partial / unsupported
    reason: str = ""
    fixed: bool = False  # revise 模式下是否已修订


@dataclass
class VerifyStats:
    checked: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    skipped: int = 0  # 带引用但因数量上限未核验的陈述
    errors: list[str] = field(default_factory=list)


def extract_claims(body: str) -> list[Claim]:
    """抽取正文中带 [n] 引用的陈述（按句；表格按行）。"""
    claims: list[Claim] = []
    seen: set[str] = set()
    in_refs = False
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            in_refs = bool(re.match(r"#+\s*(参考文献|References)", line, re.I))
            continue
        if in_refs or re.fullmatch(r"[|:\-\s]+", line):
            continue
        pieces = [line] if line.startswith("|") else [m.group(0) for m in _SENT.finditer(line)]
        for piece in pieces:
            text = re.sub(r"^\s*(?:[-*+]|\d+[.、)])\s+", "", piece).strip()
            cites = cited_numbers(text)
            if not cites or len(text) < _MIN_LEN or text in seen:
                continue
            seen.add(text)
            claims.append(Claim(id=len(claims) + 1, text=text, cites=cites))
    return claims


def sample_claims(claims: list[Claim], limit: int) -> tuple[list[Claim], int]:
    """超过上限时等距抽样，保证各章节都被覆盖。返回 (抽样结果, 被跳过数量)。"""
    if len(claims) <= limit:
        return claims, 0
    step = len(claims) / limit
    picked = [claims[int(i * step)] for i in range(limit)]
    return picked, len(claims) - len(picked)


def batch(claims: list[Claim], size: int) -> list[list[Claim]]:
    return [claims[i : i + size] for i in range(0, len(claims), size)]


def annotate_body(body: str, claims: list[Claim]) -> str:
    """在存疑陈述末尾追加标记：unsupported ⚠️，partial 🔶。"""
    for c in claims:
        if c.verdict in ("unsupported", "partial") and not c.fixed and c.text in body:
            body = body.replace(c.text, f"{c.text} {ICON[c.verdict]}", 1)
    return body


def tally(claims: list[Claim], skipped: int = 0) -> VerifyStats:
    st = VerifyStats(skipped=skipped)
    for c in claims:
        if c.verdict not in ICON:
            continue
        st.checked += 1
        setattr(st, c.verdict, getattr(st, c.verdict) + 1)
    return st


def build_appendix(claims: list[Claim], st: VerifyStats, mode: str, max_items: int = 12) -> str:
    if st.checked == 0:
        return ""
    lines = [
        "## 事实核验说明",
        "",
        f"自动核验了 {st.checked} 条带引用的陈述：✅ 支持 {st.supported} 条，"
        f"🔶 部分支持 {st.partial} 条，⚠️ 缺乏依据 {st.unsupported} 条。"
        + (f"（另有 {st.skipped} 条因数量上限未核验。）" if st.skipped else "")
        + "核验仅依据抓取到的摘要/全文节选，存疑项请对照原文确认。",
    ]
    issues = [c for c in claims if c.verdict in ("unsupported", "partial")]
    issues.sort(key=lambda c: c.verdict != "unsupported")
    if issues:
        lines += ["", "已根据来源修订以下陈述（以下为修订前的原文）：" if mode == "revise" else "存疑陈述："]
        for c in issues[:max_items]:
            text = re.sub(r"\s+", " ", c.text)
            tail = "（已修订）" if c.fixed else ""
            lines.append(f"- {ICON[c.verdict]} {text[:160]}{'…' if len(text) > 160 else ''} —— {c.reason}{tail}")
        if len(issues) > max_items:
            lines.append(f"- … 另有 {len(issues) - max_items} 条")
    return "\n".join(lines)


def sanitize_revision(revised: str, original: str, n_refs: int) -> str | None:
    """校验修订稿：引用号只能在 1..n 内（越界的删除），长度不能骤减；不合格返回 None。"""
    revised = revised.strip()
    if len(revised) < 0.5 * len(original):
        return None
    if not cited_numbers(revised):
        return None
    return remap_citations(revised, {i: i for i in range(1, n_refs + 1)})


def evidence_block(refs: list, numbers: list[int], snippet_chars: int = 1200, fulltext_chars: int = 2200) -> str:
    parts = []
    for n in numbers:
        if not 1 <= n <= len(refs):
            continue
        s = refs[n - 1]
        block = [f"[{n}] {s.get('title', '')}"]
        if s.get("snippet"):
            block.append(f"摘要/片段：{s['snippet'][:snippet_chars]}")
        if s.get("fulltext"):
            block.append(f"全文节选：{s['fulltext'][:fulltext_chars]}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)
