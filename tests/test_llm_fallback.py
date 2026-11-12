import asyncio

import pytest

from deep_research import llm as llm_mod
from deep_research.state import Reflection


class ToolChoiceLLM:
    """模拟 DeepSeek 思考模式：function calling 报 400，json_mode 正常。"""

    def __init__(self):
        self.methods = []

    def with_structured_output(self, schema, method=None):
        self.methods.append(method)
        outer = self

        class S:
            async def ainvoke(self, messages):
                if method == "function_calling":
                    raise RuntimeError("Error code: 400 - Thinking mode does not support this tool_choice")
                assert "JSON Schema" in messages[-1].content
                return schema(is_sufficient=True, knowledge_gaps="")

        return S()


def test_falls_back_to_json_mode_and_remembers(monkeypatch):
    monkeypatch.setattr(llm_mod, "_JSON_MODE_ONLY", set())
    llm = ToolChoiceLLM()
    r = asyncio.run(llm_mod.ainvoke_structured(llm, Reflection, "hi", "function_calling", "m"))
    assert r.is_sufficient and llm.methods == ["function_calling", "json_mode"]
    asyncio.run(llm_mod.ainvoke_structured(llm, Reflection, "hi", "function_calling", "m"))
    assert llm.methods == ["function_calling", "json_mode", "json_mode"]  # 第二次直接走 json_mode


def test_unrelated_errors_propagate(monkeypatch):
    monkeypatch.setattr(llm_mod, "_JSON_MODE_ONLY", set())

    class Boom:
        def with_structured_output(self, schema, method=None):
            class S:
                async def ainvoke(self, m):
                    raise RuntimeError("401 invalid api key")

            return S()

    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(llm_mod.ainvoke_structured(Boom(), Reflection, "hi", "function_calling", "m"))
