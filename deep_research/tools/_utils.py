"""工具通用辅助。"""

from __future__ import annotations

import asyncio
import weakref


class LoopLocal:
    """每个事件循环一个 Semaphore，避免跨 event loop 复用导致报错。"""

    def __init__(self, value: int = 1) -> None:
        self._value = value
        self._sems: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
            weakref.WeakKeyDictionary()
        )

    def get(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._sems.get(loop)
        if sem is None:
            sem = self._sems[loop] = asyncio.Semaphore(self._value)
        return sem
