from deep_research.verify import (
    annotate_body,
    batch,
    build_appendix,
    extract_claims,
    sample_claims,
    sanitize_revision,
    tally,
)

BODY = """# 标题
## 背景
推测解码是一种加速推理的方法 [1]。它依赖草稿模型 [1, 2]；验证阶段并行进行。没有引用的句子不会被核验。
- 列表项中的结论也需要核验才行 [3]
| 方法 | 加速比 |
|---|---|
| EAGLE 系列方法 | 3x 加速 [2] |
短句 [1]。

## 参考文献
[1] A. Paper. https://x
"""


def test_extract_claims():
    claims = extract_claims(BODY)
    texts = [c.text for c in claims]
    assert texts[0] == "推测解码是一种加速推理的方法 [1]。"
    assert texts[1] == "它依赖草稿模型 [1, 2]；"
    assert any(t.startswith("列表项中的结论") for t in texts)  # 列表标记被去掉
    assert any(t.startswith("| EAGLE") for t in texts)  # 表格按行
    assert all("没有引用" not in t and "短句" not in t and "https://x" not in t for t in texts)
    assert [c.id for c in claims] == list(range(1, len(claims) + 1))
    assert claims[1].cites == [1, 2]


def test_marker_after_period_stays_with_sentence():
    claims = extract_claims("这是一个比较长的陈述句子用于测试。[2] 另一个陈述也足够长度了呢。[3]")
    assert [c.cites for c in claims] == [[2], [3]]


def test_sample_and_batch():
    claims = extract_claims("\n".join(f"这是第{i}条足够长的陈述内容 [1]。" for i in range(30)))
    assert len(claims) == 30
    picked, skipped = sample_claims(claims, 10)
    assert len(picked) == 10 and skipped == 20
    assert picked[0] is claims[0] and picked[-1].id > 20  # 等距覆盖全文
    assert [len(b) for b in batch(claims, 12)] == [12, 12, 6]
    assert sample_claims(claims, 50) == (claims, 0)


def test_annotate_tally_appendix():
    claims = extract_claims(BODY)
    claims[0].verdict, claims[1].verdict = "supported", "unsupported"
    claims[1].reason = "来源未提及"
    claims[2].verdict, claims[2].reason = "partial", "措辞过强"
    out = annotate_body(BODY, claims)
    assert "它依赖草稿模型 [1, 2]； ⚠️" in out and "核验才行 [3] 🔶" in out
    assert "加速推理的方法 [1]。 " not in out  # supported 不加标记
    st = tally(claims, skipped=3)
    assert (st.checked, st.supported, st.partial, st.unsupported, st.skipped) == (3, 1, 1, 1, 3)
    app = build_appendix(claims, st, "annotate")
    assert "核验了 3 条" in app and "另有 3 条因数量上限未核验" in app
    assert app.index("- ⚠️ 它依赖") < app.index("- 🔶")  # 严重问题排在前面
    assert build_appendix([], tally([]), "annotate") == ""


def test_sanitize_revision():
    orig = "这是原始报告正文内容，足够长的一段话 [1]。另一段话也很长 [2]。"
    assert sanitize_revision("这是修订后的报告正文内容，足够长的一段话 [1]。另一段 [9]。", orig, 2) == (
        "这是修订后的报告正文内容，足够长的一段话 [1]。另一段 。"
    )
    assert sanitize_revision("太短 [1]", orig, 2) is None  # 长度骤减
    assert sanitize_revision("没有任何引用的修订稿，但是长度足够长足够长足够长足够长足够", orig, 2) is None
