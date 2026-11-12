"""引用编号处理：局部编号 -> 全局编号 -> 最终按出现顺序重排并生成参考文献。"""

from __future__ import annotations

import re
from typing import Iterable

from .state import Finding, Source

# 匹配 [1]、[1, 2]、[1,2,3]、[1-3]、[1，2]、【1】 等
_CITE_RE = re.compile(r"[\[【]\s*(\d+(?:\s*(?:[,，、;；]|-|–)\s*\d+)*)\s*[\]】](?!\()")


def _expand(group: str) -> list[int]:
    nums: list[int] = []
    for part in re.split(r"\s*[,，、;；]\s*", group.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a <= 50:
                nums.extend(range(a, b + 1))
                continue
            nums.extend([a, b])
        elif part.isdigit():
            nums.append(int(part))
    return nums


def cited_numbers(text: str) -> list[int]:
    """按首次出现顺序返回文本中引用的编号（去重）。"""
    seen: list[int] = []
    for m in _CITE_RE.finditer(text):
        for n in _expand(m.group(1)):
            if n not in seen:
                seen.append(n)
    return seen


def remap_citations(text: str, mapping: dict[int, int]) -> str:
    """把文本中的引用号按 mapping 替换；映射不到的编号会被删除。"""

    def repl(m: re.Match) -> str:
        new: list[int] = []
        for n in _expand(m.group(1)):
            if n in mapping and mapping[n] not in new:
                new.append(mapping[n])
        if not new:
            return ""
        return "[" + ", ".join(str(n) for n in sorted(new)) + "]"

    return _CITE_RE.sub(repl, text)


def build_global_sources(findings: Iterable[Finding]) -> tuple[list[Source], list[str]]:
    """合并所有 finding 的来源并分配全局编号，同时把各 summary 改写为全局编号。

    返回 (全局来源列表, 改写后的 summary 列表)。全局编号 = 列表下标 + 1。
    """
    registry: list[Source] = []
    index: dict[str, int] = {}
    summaries: list[str] = []
    for f in findings:
        mapping: dict[int, int] = {}
        for local_no, src in enumerate(f["sources"], start=1):
            key = src.get("key") or src.get("url") or src.get("title", "")
            if key not in index:
                registry.append(src)
                index[key] = len(registry)
            elif src.get("fulltext") and not registry[index[key] - 1].get("fulltext"):
                registry[index[key] - 1] = {**registry[index[key] - 1], "fulltext": src["fulltext"]}  # type: ignore[typeddict-item]
            mapping[local_no] = index[key]
        summaries.append(remap_citations(f["summary"], mapping))
    return registry, summaries


def renumber_report(report: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """按正文首次出现顺序将引用重新编号为 1..m，返回 (正文, 被引用的来源列表)。"""
    valid = [n for n in cited_numbers(report) if 1 <= n <= len(sources)]
    mapping = {old: new for new, old in enumerate(valid, start=1)}
    body = remap_citations(report, mapping).rstrip()
    return body, [sources[old - 1] for old in valid]


def append_references(body: str, refs: list[Source], extra_sections: str = "") -> str:
    """在正文后追加（可选的）附加章节与参考文献。"""
    out = body.rstrip()
    if extra_sections.strip():
        out += "\n\n" + extra_sections.strip()
    if refs:
        lines = ["", "", "## 参考文献", ""]
        lines += [f"[{i}] {format_reference(s)}" for i, s in enumerate(refs, start=1)]
        out += "\n".join(lines) + "\n"
    return out


def finalize_report(report: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """重新编号并附加参考文献（renumber_report + append_references）。"""
    body, refs = renumber_report(report, sources)
    return append_references(body, refs), refs


def format_reference(s: Source) -> str:
    title = s.get("title", "").strip() or s.get("url", "")
    url = s.get("url", "")
    if s.get("source_type") == "paper":
        authors = s.get("authors") or []
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        parts = [p for p in [author_str, f"*{title}*", s.get("venue", ""), s.get("published", "")[:4]] if p]
        cite = s.get("citation_count")
        text = ". ".join(parts)
        if cite:
            text += f". (被引 {cite})"
    else:
        parts = [p for p in [title, s.get("venue", ""), s.get("published", "")[:10]] if p]
        text = ". ".join(parts)
    return f"{text}. {url}" if url else text


def format_source_brief(no: int, s: Source, snippet_chars: int = 0, fulltext_chars: int = 0) -> str:
    """给 LLM 看的来源简述。"""
    meta = []
    if s.get("source_type") == "paper":
        authors = s.get("authors") or []
        if authors:
            meta.append(", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
    for k in ("venue", "published"):
        if s.get(k):
            meta.append(str(s[k])[:30])
    if s.get("citation_count"):
        meta.append(f"被引 {s['citation_count']}")
    line = f"[{no}] ({s.get('source_type', '?')}) {s.get('title', '')}"
    if meta:
        line += f" | {' | '.join(meta)}"
    if snippet_chars and s.get("snippet"):
        snippet = s["snippet"]
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + "…"
        line += f"\n    摘要/片段：{snippet}"
    if fulltext_chars and s.get("fulltext"):
        line += f"\n    全文节选：{s['fulltext'][:fulltext_chars]}"
    return line
