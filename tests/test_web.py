"""Web API 测试：任务持久化、重启恢复、取消 / 继续、报告读写。"""

import time

import pytest
from fastapi.testclient import TestClient

from deep_research.store import RunStore
from deep_research.web import server


def wait_for(client, run_id, statuses, timeout=15):
    end = time.time() + timeout
    data = {}
    while time.time() < end:
        data = client.get(f"/api/runs/{run_id}").json()
        if data["status"] in statuses:
            return data
        time.sleep(0.05)
    raise AssertionError(f"timeout, last={data.get('status')} events={data.get('events', [])[-2:]}")


@pytest.fixture
def client(fake):
    with TestClient(server.app) as c:
        yield c


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Deep Research Agent" in r.text


def test_run_with_review_verify_and_history(client):
    rid = client.post("/api/runs", json={"topic": "测试课题", "review": True, "iterations": 0}).json()["id"]
    data = wait_for(client, rid, {"waiting_review", "error"})
    assert data["status"] == "waiting_review", data["events"][-1]
    assert data["events"][-1]["type"] == "interrupt" and data["events"][-1]["plan"]

    assert client.post(f"/api/runs/{rid}/resume", json={"answer": ""}).status_code == 200
    data = wait_for(client, rid, {"done", "error"})
    assert data["status"] == "done", data["events"][-1]
    done = data["events"][-1]
    assert done["report"].startswith("# 报告") and "## 参考文献" in done["report"]
    assert done["verification"]["checked"] == 2
    assert all("fulltext" not in r for r in done["references"])  # 全文不下发给浏览器

    node_texts = [e.get("text", "") for e in data["events"] if e["type"] == "node"]
    assert any("事实核验完成" in t for t in node_texts)
    assert any("篇读取了全文" in t for t in node_texts)

    with client.stream("GET", f"/api/runs/{rid}/events") as resp:
        body = "".join(resp.iter_text())
    assert '"type": "done"' in body

    items = client.get("/api/reports").json()
    assert len(items) == 1 and items[0]["title"] == "报告"
    got = client.get(f"/api/reports/{items[0]['name']}").json()
    assert got["report"].startswith("# 报告") and got["verification"]["checked"] == 2
    assert client.get("/api/reports/..%2Fsecret.md").status_code == 404
    assert client.post(f"/api/runs/{rid}/resume", json={"answer": ""}).status_code == 409
    assert client.post(f"/api/runs/{rid}/continue").status_code == 409  # 已完成的任务不能继续


def test_report_edit(client):
    rid = client.post("/api/runs", json={"topic": "编辑测试", "iterations": 0}).json()["id"]
    wait_for(client, rid, {"done", "error"})
    name = client.get("/api/reports").json()[0]["name"]
    assert client.put(f"/api/reports/{name}", json={"report": "# 改过的标题\n正文"}).status_code == 200
    assert client.get(f"/api/reports/{name}").json()["report"].startswith("# 改过的标题")
    assert client.get("/api/reports").json()[0]["title"] == "改过的标题"
    assert client.put("/api/reports/nope.md", json={"report": "x"}).status_code == 404
    assert client.put(f"/api/reports/{name}", json={"report": ""}).status_code == 422


def test_error_then_continue_resumes_from_checkpoint(client, fake):
    fake.fail_research = 3  # 让研究节点失败（含自动重试）
    rid = client.post("/api/runs", json={"topic": "测试课题", "iterations": 0, "verify": "off"}).json()["id"]
    data = wait_for(client, rid, {"error", "done"})
    assert data["status"] == "error"
    assert "继续" in data["events"][-1]["message"]
    assert len(fake.plan_calls) == 1

    fake.fail_research = 0
    assert client.post(f"/api/runs/{rid}/continue").status_code == 200
    data = wait_for(client, rid, {"done", "error"})
    assert data["status"] == "done", data["events"][-1]
    assert len(fake.plan_calls) == 1  # 规划没有重跑，从失败处继续
    assert any(e.get("node") == "resume" for e in data["events"])


def test_cancel_then_continue(client, fake):
    fake.research_delay = 5
    rid = client.post("/api/runs", json={"topic": "测试课题", "iterations": 0, "verify": "off"}).json()["id"]
    end = time.time() + 10
    while time.time() < end:  # 等待规划完成、研究开始
        if any(e.get("node") == "plan" for e in client.get(f"/api/runs/{rid}").json()["events"]):
            break
        time.sleep(0.05)
    time.sleep(0.3)
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 200
    data = wait_for(client, rid, {"cancelled"})
    assert data["events"][-1]["type"] == "cancelled"
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 409

    fake.research_delay = 0
    assert client.post(f"/api/runs/{rid}/continue").status_code == 200
    data = wait_for(client, rid, {"done", "error"})
    assert data["status"] == "done", data["events"][-1]
    assert len(fake.plan_calls) == 1


def test_persistence_across_restart(fake):
    with TestClient(server.app) as c:
        rid = c.post("/api/runs", json={"topic": "重启测试", "review": True, "iterations": 0, "verify": "off"}).json()["id"]
        wait_for(c, rid, {"waiting_review"})
        done_id = c.post("/api/runs", json={"topic": "另一个", "iterations": 0, "verify": "off"}).json()["id"]
        wait_for(c, done_id, {"done"})

    # 模拟"服务异常退出时任务仍在运行"
    store = RunStore(server.DATA_DIR / "runs.db")
    killed = "deadbeef0001"
    store.create_run(killed, "被中断的任务", {"iterations": 0}, "running", "2026-01-01T00:00:00")
    store.close()

    with TestClient(server.app) as c:  # "重启"
        runs = {r["id"]: r for r in c.get("/api/runs").json()}
        assert runs[rid]["status"] == "waiting_review"  # 审核中的任务保持等待
        assert runs[done_id]["status"] == "done"
        assert runs[killed]["status"] == "interrupted"  # 运行中被打断的任务
        # 历史事件已持久化并可重放
        events = c.get(f"/api/runs/{done_id}").json()["events"]
        assert events[-1]["type"] == "done" and events[-1]["report"].startswith("# 报告")
        # 重启后仍能回应计划审核，继续跑完
        assert c.post(f"/api/runs/{rid}/resume", json={"answer": ""}).status_code == 200
        data = wait_for(c, rid, {"done", "error"})
        assert data["status"] == "done", data["events"][-1]
        # 没有断点的中断任务：continue 会从头开始
        assert c.post(f"/api/runs/{killed}/continue").status_code == 200
        data = wait_for(c, killed, {"done", "error"})
        assert data["status"] == "done", data["events"][-1]


def test_delete_run(client):
    rid = client.post("/api/runs", json={"topic": "删除测试", "iterations": 0, "verify": "off"}).json()["id"]
    wait_for(client, rid, {"done"})
    assert client.delete(f"/api/runs/{rid}").status_code == 200
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert rid not in [r["id"] for r in client.get("/api/runs").json()]


def test_concurrency_limit_queues_runs(fake, monkeypatch):
    monkeypatch.setattr(server, "MAX_CONCURRENT_RUNS", 1)
    fake.research_delay = 0.6
    with TestClient(server.app) as c:
        a = c.post("/api/runs", json={"topic": "任务A", "iterations": 0, "verify": "off"}).json()["id"]
        time.sleep(0.2)
        b = c.post("/api/runs", json={"topic": "任务B", "iterations": 0, "verify": "off"}).json()["id"]
        time.sleep(0.2)
        assert c.get(f"/api/runs/{b}").json()["status"] == "queued"
        wait_for(c, a, {"done"})
        wait_for(c, b, {"done"})
        evs = c.get(f"/api/runs/{b}").json()["events"]
        assert evs[0]["type"] == "queued"


def test_validation(client):
    assert client.post("/api/runs", json={"topic": "x"}).status_code == 422
    assert client.post("/api/runs", json={"topic": "合法课题", "verify": "bogus"}).status_code == 422
    assert client.get("/api/runs/nope").status_code == 404


def test_upload_doc_and_copilot_endpoints(client):
    import base64

    b64 = base64.b64encode("这是一份关于推测解码本地白皮书的核心内容。".encode("utf-8")).decode("ascii")
    r_up = client.post("/api/upload-doc", json={"filename": "spec_whitepaper.md", "content_base64": b64})
    assert r_up.status_code == 200
    doc = r_up.json()
    assert doc["origin"] == "local" and "spec whitepaper" in doc["title"]
    assert doc["has_fulltext"] is True and "推测解码" in doc["fulltext"]

    # Run with local_docs attached
    rid = client.post(
        "/api/runs",
        json={"topic": "混合调研测试", "iterations": 0, "verify": "off", "local_docs": [doc]},
    ).json()["id"]
    data = wait_for(client, rid, {"done", "error"})
    assert data["status"] == "done"

    # Copilot Ask endpoint
    r_ask = client.post(
        "/api/copilot/ask",
        json={"question": "核心结论是什么？", "report": "# 报告\n推测解码加速推理 [1]。", "references": [doc]},
    )
    assert r_ask.status_code == 200 and r_ask.json()["answer"]

    # Copilot Ask Stream (SSE) endpoint
    r_stream = client.post(
        "/api/copilot/ask-stream",
        json={"question": "流式追问核心结论？", "report": "# 报告\n推测解码加速推理 [1]。", "references": [doc]},
    )
    assert r_stream.status_code == 200
    assert "text/event-stream" in r_stream.headers.get("content-type", "")
    assert '"done": true' in r_stream.text

    # Section Rewrite endpoint
    r_rw = client.post(
        "/api/copilot/rewrite-section",
        json={
            "topic": "推测解码",
            "section_markdown": "## 2. 技术路线\n推测解码包含草稿模型与验证 [1]。",
            "instruction": "深入扩写细节",
            "references": [doc],
        },
    )
    assert r_rw.status_code == 200 and r_rw.json()["markdown"]
