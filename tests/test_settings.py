import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from deep_research import llm as llm_mod
from deep_research import settings
from deep_research.web import server

ENV_VARS = [e for e, _ in settings.FIELDS.values()]


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("SETTINGS_PATH", str(path))
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    # 以"干净环境"作为原始值，并在测试后还原 os.environ
    monkeypatch.setattr(settings, "_ORIGINAL", {v: None for v in ENV_VARS})
    with TestClient(server.app) as c:
        c.settings_path = path
        yield c
    for v in ENV_VARS:
        monkeypatch.delenv(v, raising=False)


def test_secret_masked_and_persisted(client):
    view = client.get("/api/settings").json()
    assert view["openai_api_key"] == {"set": False, "preview": "", "source": "unset"}

    r = client.put(
        "/api/settings",
        json={"values": {"openai_api_key": "sk-secret-1234567890", "llm_model": "gpt-x", "openai_base_url": "http://h/v1"}},
    )
    assert r.status_code == 200
    body = r.text
    assert "sk-secret-1234567890" not in body
    view = r.json()
    assert view["openai_api_key"]["set"] and view["openai_api_key"]["source"] == "ui"
    assert view["llm_model"]["value"] == "gpt-x"

    assert json.loads(client.settings_path.read_text(encoding="utf-8"))["openai_api_key"] == "sk-secret-1234567890"
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-secret-1234567890"
    assert os.environ["OPENAI_BASE_URL"] == "http://h/v1"


def test_blank_secret_keeps_value_and_blank_text_clears(client):
    client.put("/api/settings", json={"values": {"openai_api_key": "sk-aaaaaaaaaaaa", "llm_model": "m1"}})
    view = client.put("/api/settings", json={"values": {"openai_api_key": "", "llm_model": ""}}).json()
    assert view["openai_api_key"]["set"]  # 空密钥 = 保持不变
    assert view["llm_model"]["source"] == "unset"  # 非密钥空串 = 清除
    view = client.put("/api/settings", json={"values": {}, "clear": ["openai_api_key"]}).json()
    assert not view["openai_api_key"]["set"]


def test_unknown_field_rejected(client):
    assert client.put("/api/settings", json={"values": {"evil": "x"}}).status_code == 422


def test_connection_test(client, monkeypatch):
    # 未配置 Key
    r = client.post("/api/settings/test").json()
    assert r["ok"] is False and "OPENAI_API_KEY" in r["error"]

    class Fake:
        async def ainvoke(self, _):
            return AIMessage("ok")

    client.put("/api/settings", json={"values": {"openai_api_key": "sk-aaaaaaaaaaaa", "llm_model": "m1"}})
    monkeypatch.setattr(server, "get_llm", lambda *a, **k: Fake())
    r = client.post("/api/settings/test").json()
    assert r == {"ok": True, "model": "m1", "reply": "ok"}


def test_llm_cache_follows_key_change(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-one")
    a = llm_mod.get_llm("gpt-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-two")
    b = llm_mod.get_llm("gpt-x")
    assert a is not b and llm_mod.get_llm("gpt-x") is b
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError):
        llm_mod.get_llm("gpt-x")


def test_admin_token_locks_settings_when_configured(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "my-super-admin-pass")
    view = client.get("/api/settings").json()
    assert view["admin_locked"] is True

    # Reject without token or with wrong token
    r_bad = client.put("/api/settings", json={"values": {"llm_model": "hacked"}})
    assert r_bad.status_code == 403

    # Succeed with valid admin_token
    r_ok = client.put(
        "/api/settings",
        json={"values": {"llm_model": "authorized-model"}, "admin_token": "my-super-admin-pass"},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["llm_model"]["value"] == "authorized-model"
