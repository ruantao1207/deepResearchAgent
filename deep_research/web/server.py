"""Web 服务：FastAPI + SSE，任务与执行状态均持久化（服务重启后可断点续跑）。

GET    /api/runs                    任务列表
POST   /api/runs                    创建调研任务
GET    /api/runs/{id}               任务快照（状态 + 全部事件）
GET    /api/runs/{id}/events        SSE 实时事件流（支持 last_event_id / Last-Event-ID 续传）
POST   /api/runs/{id}/resume        回应计划审核
POST   /api/runs/{id}/cancel        取消运行中的任务（保留断点）
POST   /api/runs/{id}/continue      从断点继续（已取消 / 出错 / 服务重启中断的任务）
DELETE /api/runs/{id}               删除任务记录与断点
GET    /api/reports[/{name}]        历史报告；PUT /api/reports/{name} 保存编辑
GET/PUT /api/settings               模型与 API 设置；POST /api/settings/test 连接测试
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from .. import nodes, settings
from ..citations import format_source_brief
from ..cli import _describe
from ..config import Configuration
from ..graph import build_graph
from ..llm import get_llm
from ..prompts import COPILOT_ASK_PROMPT, SECTION_REWRITE_PROMPT
from ..reports import save_report, strip_fulltext
from ..store import RunStore

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(os.getenv("DATA_DIR", ".deep_research"))
REPORTS_DIR = Path("reports")
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "2"))

TERMINAL = {"done", "error", "cancelled", "interrupted"}
CONTINUABLE = {"cancelled", "interrupted", "error"}

settings.apply()
CTX = SimpleNamespace(graph=None, store=None, saver=None, sem=None)
RUNS: dict[str, "Run"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CTX.store = RunStore(DATA_DIR / "runs.db")
    CTX.sem = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
    RUNS.clear()
    async with AsyncSqliteSaver.from_conn_string(str(DATA_DIR / "checkpoints.db")) as saver:
        CTX.saver = saver
        CTX.graph = build_graph(checkpointer=saver)
        for rec in CTX.store.list_runs():
            run = Run.from_record(rec)
            if run.status in ("running", "queued"):  # 上次服务退出时尚未跑完
                run.status = "interrupted"
                CTX.store.set_status(run.id, "interrupted")
            RUNS[run.id] = run
        try:
            yield
        finally:
            for run in RUNS.values():
                if run.task and not run.task.done():
                    run.task.cancel()
            await asyncio.gather(
                *(r.task for r in RUNS.values() if r.task), return_exceptions=True
            )
    CTX.store.close()


app = FastAPI(title="Deep Research Agent", lifespan=lifespan)


class RunRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=500)
    review: bool = False
    model: str | None = None
    writer_model: str | None = None
    max_sub_questions: int | None = Field(None, ge=1, le=12)
    iterations: int | None = Field(None, ge=0, le=5)
    language: str | None = None
    verify: Literal["off", "annotate", "revise"] | None = None
    fulltext: bool | None = None
    relevance_filter: bool | None = None
    local_docs: list[dict[str, Any]] = Field(default_factory=list)


class ResumeRequest(BaseModel):
    answer: str = ""


def _configurable(opts: dict[str, Any], run_id: str) -> dict[str, Any]:
    mapping = {
        "llm_model": opts.get("model") or None,
        "writer_model": opts.get("writer_model") or None,
        "max_sub_questions": opts.get("max_sub_questions"),
        "max_iterations": opts.get("iterations"),
        "report_language": opts.get("language") or None,
        "verify_mode": opts.get("verify"),
        "enable_fulltext": opts.get("fulltext"),
        "enable_relevance_filter": opts.get("relevance_filter"),
    }
    return {
        "thread_id": run_id,
        "human_review": bool(opts.get("review")),
        **{k: v for k, v in mapping.items() if v is not None},
    }


class Run:
    def __init__(
        self,
        run_id: str,
        topic: str,
        options: dict[str, Any],
        status: str,
        created: str,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.id, self.topic, self.options = run_id, topic, options
        self.status, self.created = status, created
        self._events = events
        self.live_draft: str = ""
        self.task: asyncio.Task | None = None
        self.cond = asyncio.Condition()
        self.config = {"configurable": _configurable(options, run_id), "recursion_limit": 100}

    # ---- 构造 ----
    @classmethod
    def create(cls, req: RunRequest) -> "Run":
        run = cls(
            uuid.uuid4().hex[:12],
            req.topic,
            req.model_dump(),
            "running",
            datetime.now().isoformat(timespec="seconds"),
            events=[],
        )
        CTX.store.create_run(run.id, run.topic, run.options, run.status, run.created)
        return run

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Run":
        return cls(rec["id"], rec["topic"], rec["options"], rec["status"], rec["created"], events=None)

    @property
    def events(self) -> list[dict[str, Any]]:
        if self._events is None:  # 懒加载，避免启动时读取所有历史事件
            self._events = CTX.store.load_events(self.id)
        return self._events

    # ---- 状态 / 事件 ----
    def set_status(self, status: str) -> None:
        self.status = status
        CTX.store.set_status(self.id, status)

    async def emit(self, **event: Any) -> None:
        async with self.cond:
            event["id"] = len(self.events)
            self.events.append(event)
            CTX.store.add_event(self.id, event["id"], event)
            self.cond.notify_all()

    async def on_draft(self, text: str) -> None:
        async with self.cond:
            self.live_draft = text
            self.cond.notify_all()

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "topic": self.topic, "status": self.status, "created": self.created}

    # ---- 执行 ----
    async def _has_checkpoint(self) -> bool:
        snap = await CTX.graph.aget_state(self.config)
        return bool(snap.values) or bool(snap.next)

    async def drive(self, payload: object) -> None:
        graph = CTX.graph
        interrupted = None
        try:
            if CTX.sem.locked():
                self.set_status("queued")
                await self.emit(type="queued", text="⏳ 排队中：并发任务已满，稍后自动开始…")
            async with CTX.sem:
                self.set_status("running")
                if payload is None:
                    if await self._has_checkpoint():
                        await self.emit(type="node", node="resume", text="▶️ 从断点继续…")
                    else:
                        payload = {
                            "topic": self.topic,
                            "local_docs": self.options.get("local_docs") or [],
                        }
                tok = nodes.STREAM_CALLBACK.set(self.on_draft)
                try:
                    async for chunk in graph.astream(payload, self.config, stream_mode="updates"):
                        for node, update in chunk.items():
                            if node == "__interrupt__":
                                interrupted = update[0].value
                            else:
                                await self.emit(
                                    type="node",
                                    node=node,
                                    text=_describe(node, update),
                                    meta=strip_fulltext(update) if isinstance(update, dict) else {},
                                )
                finally:
                    nodes.STREAM_CALLBACK.reset(tok)
            if interrupted is not None:
                self.set_status("waiting_review")
                await self.emit(type="interrupt", **{k: v for k, v in interrupted.items() if k != "type"})
                return
            state = (await graph.aget_state(self.config)).values
            report = state.get("report", "")
            if not report:
                raise RuntimeError("未生成报告")
            path = save_report(REPORTS_DIR, self.topic, state, report)
            self.set_status("done")
            await self.emit(
                type="done",
                report=report,
                research_type=state.get("research_type"),
                plan=state.get("plan", []),
                findings=strip_fulltext(state.get("findings", [])),
                references=strip_fulltext(state.get("references", [])),
                verification=state.get("verification") or {},
                saved_as=path.name,
            )
        except asyncio.CancelledError:
            self.set_status("cancelled")
            await self.emit(type="cancelled", message="已取消。已完成的步骤已保存，可点击「继续」从断点恢复。")
        except Exception as e:  # noqa: BLE001
            self.set_status("error")
            await self.emit(type="error", message=f"{type(e).__name__}: {e}（修复后可点击「继续」从失败处重试）")

    def start(self, payload: object) -> None:
        self.task = asyncio.create_task(self.drive(payload))


def _get_run(run_id: str) -> Run:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


# ---------------------------------------------------------------------------
# 任务接口
# ---------------------------------------------------------------------------


@app.get("/api/runs")
async def list_runs():
    return sorted((r.summary() for r in RUNS.values()), key=lambda r: r["created"], reverse=True)


@app.post("/api/runs")
async def create_run(req: RunRequest):
    run = Run.create(req)
    RUNS[run.id] = run
    run.start({"topic": req.topic, "local_docs": req.local_docs})
    return {"id": run.id}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = _get_run(run_id)
    return {**run.summary(), "options": run.options, "events": run.events}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, body: ResumeRequest):
    run = _get_run(run_id)
    if run.status != "waiting_review":
        raise HTTPException(409, "run is not waiting for review")
    run.start(Command(resume=body.answer))
    return {"ok": True}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    run = _get_run(run_id)
    if run.task and not run.task.done():
        run.task.cancel()
        return {"ok": True}
    if run.status == "waiting_review":
        run.set_status("cancelled")
        await run.emit(type="cancelled", message="已取消。可点击「继续」重新审核计划。")
        return {"ok": True}
    raise HTTPException(409, f"run is not cancellable (status={run.status})")


@app.post("/api/runs/{run_id}/continue")
async def continue_run(run_id: str):
    run = _get_run(run_id)
    if run.status not in CONTINUABLE:
        raise HTTPException(409, f"run cannot be continued (status={run.status})")
    run.start(None)
    return {"ok": True}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    run = _get_run(run_id)
    if run.task and not run.task.done():
        run.task.cancel()
        await asyncio.gather(run.task, return_exceptions=True)
    CTX.store.delete_run(run_id)
    RUNS.pop(run_id, None)
    try:
        await CTX.saver.adelete_thread(run_id)
    except Exception:  # noqa: BLE001  断点清理失败不影响删除
        pass
    return {"ok": True}


@app.get("/api/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    last_event_id: int | None = None,
    last_event_id_header: str | None = Header(None, alias="Last-Event-ID"),
):
    run = _get_run(run_id)
    if last_event_id is None and last_event_id_header and last_event_id_header.lstrip("-").isdigit():
        last_event_id = int(last_event_id_header)
    start = (last_event_id if last_event_id is not None else -1) + 1

    async def gen():
        idx = start
        last_draft = ""
        while True:
            async with run.cond:
                try:
                    await asyncio.wait_for(
                        run.cond.wait_for(
                            lambda: idx < len(run.events)
                            or run.live_draft != last_draft
                            or run.status in TERMINAL
                        ),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                batch = run.events[idx:]
                cur_draft = run.live_draft
            if cur_draft and cur_draft != last_draft:
                last_draft = cur_draft
                yield f"data: {json.dumps({'type': 'draft', 'text': cur_draft}, ensure_ascii=False)}\n\n"
            for ev in batch:
                yield f"id: {ev['id']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            idx += len(batch)
            if run.status in TERMINAL and idx >= len(run.events):
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 报告接口
# ---------------------------------------------------------------------------


def _report_path(name: str) -> Path:
    path = (REPORTS_DIR / Path(name).name).resolve()
    if path.suffix != ".md" or path.parent != REPORTS_DIR.resolve() or not path.exists():
        raise HTTPException(404, "report not found")
    return path


@app.get("/api/reports")
async def list_reports():
    if not REPORTS_DIR.exists():
        return []
    items = []
    for p in sorted(REPORTS_DIR.glob("*.md"), reverse=True):
        first = p.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
        items.append({"name": p.name, "title": first or p.stem})
    return items


@app.get("/api/reports/{name}")
async def get_report(name: str):
    path = _report_path(name)
    meta: dict[str, Any] = {}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    return {
        "name": path.name,
        "report": path.read_text(encoding="utf-8"),
        "topic": meta.get("topic") or "",
        "research_type": meta.get("research_type") or "",
        "plan": meta.get("plan") or [],
        "findings": meta.get("findings") or [],
        "references": meta.get("references") or [],
        "verification": meta.get("verification") or {},
    }


class ReportUpdate(BaseModel):
    report: str = Field(min_length=1, max_length=2_000_000)


@app.put("/api/reports/{name}")
async def put_report(name: str, body: ReportUpdate):
    path = _report_path(name)
    path.write_text(body.report, encoding="utf-8")
    return {"ok": True}


# ---------------------------------------------------------------------------
# 本地文档上传解析 & 伴读 Copilot / 章节局部重写接口
# ---------------------------------------------------------------------------


class UploadDocRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=20_000_000)


@app.post("/api/upload-doc")
async def upload_doc(body: UploadDocRequest):
    """解析用户上传的本地 PDF / Markdown / TXT 文档，返回标准 Source 结构。"""
    try:
        raw = base64.b64decode(body.content_base64)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"invalid base64 content: {e}") from e

    fname = Path(body.filename).name
    ext = Path(fname).suffix.lower()
    text = ""
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages[:40]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                continue
        text = "\n\n".join(p.strip() for p in pages if p.strip())
    else:
        for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue

    text = " ".join(text.split()) if ext == ".pdf" else text.strip()
    if not text:
        raise HTTPException(422, f"未能从文件 {fname} 中提取出有效文本")

    title = Path(fname).stem.replace("_", " ").strip() or fname
    snippet = text[:600]
    fulltext = text[:16000]
    return {
        "key": f"local:{fname}",
        "source_type": "paper" if ext == ".pdf" else "web",
        "origin": "local",
        "title": f"[本地文档] {title}",
        "url": f"local://{fname}",
        "snippet": snippet,
        "authors": ["本地上传资料"],
        "published": datetime.now().strftime("%Y-%m-%d"),
        "venue": f"本地文件 ({ext.lstrip('.').upper()})",
        "relevance": 10.0,
        "fulltext": fulltext,
        "has_fulltext": True,
    }


class CopilotAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    report: str = Field(min_length=1, max_length=500_000)
    references: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)


def _build_copilot_prompt(body: CopilotAskRequest) -> str:
    sources_text = "\n".join(
        format_source_brief(i, s, snippet_chars=350)  # type: ignore[arg-type]
        for i, s in enumerate(body.references[:60], start=1)
    )
    hist_lines = []
    for h in body.history[-6:]:
        role = "用户" if h.get("role") == "user" else "助手"
        hist_lines.append(f"{role}：{h.get('content', '')[:600]}")
    return COPILOT_ASK_PROMPT.format(
        report=body.report[:35000],
        sources=sources_text or "（详见报告文末参考文献）",
        history="\n".join(hist_lines) or "（无历史对话）",
        question=body.question.strip(),
    )


@app.post("/api/copilot/ask")
async def copilot_ask(body: CopilotAskRequest):
    """基于当前报告与引用文献进行伴读问答（Ask This Report）。"""
    cfg = Configuration.from_runnable_config({})
    prompt = _build_copilot_prompt(body)
    llm = get_llm(cfg.llm_model, cfg.temperature)
    resp = await llm.ainvoke([HumanMessage(prompt)])
    return {"answer": nodes._text(resp).strip()}


@app.post("/api/copilot/ask-stream")
async def copilot_ask_stream(body: CopilotAskRequest):
    """基于当前报告与引用文献进行 SSE 逐字流式伴读问答。"""
    cfg = Configuration.from_runnable_config({})
    prompt = _build_copilot_prompt(body)
    llm = get_llm(cfg.llm_model, cfg.temperature)

    async def gen():
        chunks: list[str] = []
        try:
            if hasattr(llm, "astream") and type(llm).__name__ != "FakeLLM":
                async for chunk in llm.astream([HumanMessage(prompt)]):
                    piece = nodes._text(chunk)
                    if piece:
                        chunks.append(piece)
                        yield f"data: {json.dumps({'chunk': piece}, ensure_ascii=False)}\n\n"
            else:
                resp = await llm.ainvoke([HumanMessage(prompt)])
                full = nodes._text(resp).strip()
                chunks.append(full)
                yield f"data: {json.dumps({'chunk': full}, ensure_ascii=False)}\n\n"
            answer = "".join(chunks).strip()
            yield f"data: {json.dumps({'done': True, 'answer': answer}, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class SectionRewriteRequest(BaseModel):
    topic: str = ""
    section_markdown: str = Field(min_length=5, max_length=100_000)
    instruction: str = Field(min_length=2, max_length=1000)
    references: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/api/copilot/rewrite-section")
async def rewrite_section(body: SectionRewriteRequest):
    """根据用户指令对单个 ## 章节进行局部 AI 重写/扩写/表格化，保持引用编号对齐。"""
    cfg = Configuration.from_runnable_config({})
    sources_text = "\n".join(
        format_source_brief(i, s, snippet_chars=300)  # type: ignore[arg-type]
        for i, s in enumerate(body.references[:60], start=1)
    )
    prompt = SECTION_REWRITE_PROMPT.format(
        topic=body.topic or "深度调研报告",
        sources=sources_text or "（请保持原章节中的 [n] 引用号不变）",
        section_markdown=body.section_markdown.strip(),
        instruction=body.instruction.strip(),
    )
    writer = get_llm(cfg.effective_writer_model, cfg.temperature)
    resp = await writer.ainvoke([HumanMessage(prompt)])
    out = nodes._text(resp).strip()
    if out.startswith("```"):
        lines = out.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        out = "\n".join(lines).strip()
    return {"markdown": out}


# ---------------------------------------------------------------------------
# 设置接口
# ---------------------------------------------------------------------------


@app.get("/api/settings")
async def read_settings():
    view = settings.public_view()
    view["admin_locked"] = bool(os.getenv("ADMIN_TOKEN", "").strip())
    return view


class SettingsUpdate(BaseModel):
    values: dict[str, str | None] = Field(default_factory=dict)
    clear: list[str] = Field(default_factory=list)
    admin_token: str | None = None


@app.put("/api/settings")
async def write_settings(body: SettingsUpdate):
    expected_token = os.getenv("ADMIN_TOKEN", "").strip()
    if expected_token:
        import hmac

        provided = (body.admin_token or "").strip()
        if not provided or not hmac.compare_digest(provided, expected_token):
            raise HTTPException(403, "服务端已开启管理员保护（ADMIN_TOKEN），请输入正确的管理员口令后再修改配置。")
    unknown = [k for k in [*body.values, *body.clear] if k not in settings.FIELDS]
    if unknown:
        raise HTTPException(422, f"unknown settings: {unknown}")
    settings.update(body.values, body.clear)
    view = settings.public_view()
    view["admin_locked"] = bool(expected_token)
    return view


@app.post("/api/settings/test")
async def test_settings():
    """用当前生效的配置发一个最小请求，验证 Key / Base URL / 模型是否可用。"""
    cfg = Configuration.from_runnable_config({})
    try:
        llm = get_llm(cfg.llm_model, cfg.temperature)
        resp = await asyncio.wait_for(llm.ainvoke("Reply with the single word: ok"), timeout=45)
        text = str(getattr(resp, "content", ""))[:80]
        return {"ok": True, "model": cfg.llm_model, "reply": text}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": cfg.llm_model, "error": f"{type(e).__name__}: {str(e)[:300]}"}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Deep Research Agent Web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    settings.apply()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
