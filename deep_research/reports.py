"""报告落盘：Markdown 报告 + JSON 中间结果（计划 / 研究笔记 / 参考来源 / 核验结果）。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def strip_fulltext(obj: Any) -> Any:
    """递归去掉来源里体积较大的 fulltext 字段，仅保留 has_fulltext 标记。"""
    if isinstance(obj, dict):
        out = {k: strip_fulltext(v) for k, v in obj.items() if k != "fulltext"}
        if obj.get("fulltext"):
            out["has_fulltext"] = True
        return out
    if isinstance(obj, list):
        return [strip_fulltext(x) for x in obj]
    return obj


def save_report(reports_dir: Path, topic: str, state: dict, report: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic).strip("_")[:40]
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
    md = reports_dir / f"{stem}.md"
    md.write_text(report, encoding="utf-8")
    meta = {
        "topic": topic,
        "research_type": state.get("research_type"),
        "plan": state.get("plan"),
        "findings": strip_fulltext(state.get("findings")),
        "references": strip_fulltext(state.get("references")),
        "verification": state.get("verification"),
    }
    (reports_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return md
