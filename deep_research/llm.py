"""LLM 工厂：兼容任何 OpenAI 协议的服务（OpenAI / DeepSeek / Qwen / vLLM ...）。

Key / Base URL 在每次调用时从环境变量读取，并参与缓存键，
因此在 Web 界面修改设置后无需重启即可生效。
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=16)
def _build(model: str, temperature: float | None, api_key: str | None, base_url: str | None) -> BaseChatModel:
    kwargs: dict = {"model": model, "max_retries": 3, "timeout": 180}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def get_llm(model: str, temperature: float | None = None) -> BaseChatModel:
    api_key = os.getenv("OPENAI_API_KEY") or None
    if not api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY：请在 Web 界面的「设置」中填写，或写入 .env")
    return _build(model, temperature, api_key, os.getenv("OPENAI_BASE_URL") or None)


# ---------------------------------------------------------------------------
# 结构化输出：优先 function calling，不被支持时自动降级为 JSON 模式
# ---------------------------------------------------------------------------

# 已确认不支持 function calling（强制 tool_choice）的 (base_url, model)
_JSON_MODE_ONLY: set[tuple[str, str]] = set()

_JSON_HINT = (
    "\n\n---\n请只输出一个符合下列 JSON Schema 的 JSON 对象，不要输出任何其他文字或 Markdown 代码块：\n{schema}"
)


def _needs_fallback(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("tool_choice", "tools", "function call", "function_call", "does not support", "unsupported")
    )


async def ainvoke_structured(llm, schema, prompt: str, method: str = "function_calling", model: str = ""):
    """调用 LLM 并解析为 Pydantic `schema`。

    method=function_calling 失败且错误提示不支持工具调用时（例如 DeepSeek 思考模式不支持
    tool_choice），自动改用 json_mode，并记住该模型避免重复试错。
    """
    from langchain_core.messages import HumanMessage

    ident = (os.getenv("OPENAI_BASE_URL") or "", model)
    if method != "json_mode" and ident not in _JSON_MODE_ONLY:
        try:
            return await llm.with_structured_output(schema, method=method).ainvoke([HumanMessage(prompt)])
        except Exception as e:  # noqa: BLE001
            if not _needs_fallback(e):
                raise
            _JSON_MODE_ONLY.add(ident)
    import json

    hinted = prompt + _JSON_HINT.format(schema=json.dumps(schema.model_json_schema(), ensure_ascii=False))
    return await llm.with_structured_output(schema, method="json_mode").ainvoke([HumanMessage(hinted)])
