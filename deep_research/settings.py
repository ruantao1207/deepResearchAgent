"""可在 Web 界面中修改的运行时设置（LLM / 搜索 API）。

- 持久化到本地 JSON（默认 .deep_research_settings.json，可用 SETTINGS_PATH 指定），已加入 .gitignore
- 保存后立即写入 os.environ，下次调用即生效（llm.py / tools / config.py 都是调用时读取环境变量）
- 密钥永不返回给前端，只给脱敏预览；留空表示保持不变，需显式 clear 才会删除
- 清除某项设置后恢复为启动时（.env / 系统环境变量）的原始值
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import config as _config  # noqa: F401  确保 .env 已加载

# 字段名 -> (环境变量, 是否为密钥)
FIELDS: dict[str, tuple[str, bool]] = {
    "openai_base_url": ("OPENAI_BASE_URL", False),
    "openai_api_key": ("OPENAI_API_KEY", True),
    "llm_model": ("LLM_MODEL", False),
    "writer_model": ("WRITER_MODEL", False),
    "tavily_api_key": ("TAVILY_API_KEY", True),
    "s2_api_key": ("S2_API_KEY", True),
    "github_token": ("GITHUB_TOKEN", True),
}

# 启动时（.env / 系统环境）的原始值，用于"清除设置"后恢复
_ORIGINAL: dict[str, str | None] = {env: os.environ.get(env) for env, _s in FIELDS.values()}


def settings_path() -> Path:
    return Path(os.getenv("SETTINGS_PATH", ".deep_research_settings.json"))


def _read() -> dict[str, str]:
    p = settings_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: str(v) for k, v in data.items() if k in FIELDS and v}


def _write(data: dict[str, str]) -> None:
    p = settings_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        pass


def apply() -> None:
    """把已保存的设置应用到环境变量；未设置的项恢复为原始值。"""
    saved = _read()
    for name, (env, _secret) in FIELDS.items():
        if saved.get(name):
            os.environ[env] = saved[name]
        elif _ORIGINAL[env] is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = _ORIGINAL[env]  # type: ignore[assignment]


def mask(value: str) -> str:
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:3]}…{value[-4:]}"


def public_view() -> dict[str, Any]:
    """当前生效配置的前端视图（密钥脱敏）。"""
    saved = _read()
    out: dict[str, Any] = {}
    for name, (env, secret) in FIELDS.items():
        current = os.environ.get(env, "")
        source = "ui" if name in saved else ("env" if current else "unset")
        if secret:
            out[name] = {"set": bool(current), "preview": mask(current) if current else "", "source": source}
        else:
            out[name] = {"value": current, "source": source}
    return out


def update(values: dict[str, str | None], clear: list[str] | None = None) -> None:
    """合并更新。None/空串 = 保持不变（密钥）；非密钥字段空串表示清除覆盖。"""
    saved = _read()
    for name in clear or []:
        saved.pop(name, None)
    for name, val in values.items():
        if name not in FIELDS:
            continue
        _env, secret = FIELDS[name]
        if val is None:
            continue
        val = val.strip()
        if val:
            saved[name] = val
        elif not secret:
            saved.pop(name, None)
    _write(saved)
    apply()
