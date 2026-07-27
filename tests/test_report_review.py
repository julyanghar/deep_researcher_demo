"""REPORT_REVIEW(审阅+补节)机制单测:A1-A8 见 modify-code-runs/report-review-supplement/blueprint.md。

区分度自检:先在基线代码上跑——test_off_matches_baseline/test_summary_path_unchanged
应 pass(golden 即基线行为),其余应 fail(机制不存在)。实现后应全 pass。
"""

import asyncio
import json

import pytest

from deep_researcher_demo.agents import FinalWriter, outline_then_parallel


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i + len(start):j] if j >= 0 else ""


OUTLINE_JSON = json.dumps({
    "sections": [
        {"title": "Alpha", "source_ids": [1]},
        {"title": "Beta", "source_ids": [2]},
        {"title": "Gamma", "source_ids": [1, 2]},
    ]
})

# 基线 golden(REPORT_PARALLEL=1,无 review):outline 三节 → 各节体 → "## 题\n\n体"拼接 + References
BASELINE_BODY = (
    "## Alpha\n\nBody of Alpha.\n\n"
    "## Beta\n\nBody of Beta.\n\n"
    "## Gamma\n\nBody of Gamma."
)
BASELINE_REPORT = BASELINE_BODY + "\n\n## References\n\n- http://u1\n- http://u2"


class ReviewStubClient:
    """tag 分发 stub:记录 (seq, tag),可配置审阅 JSON/补节失败集合。"""

    def __init__(self, review=None, addition_fail_titles=()):
        self.review = review  # None=不该被调 / "GARBAGE" / dict
        self.addition_fail_titles = set(addition_fail_titles)
        self.calls = []  # [(seq, tag)]
        self._seq = 0

    async def chat(self, messages, *, model, temperature=0.0, max_tokens=None,
                   store_generated_kv=False, tag=None):
        self._seq += 1
        self.calls.append((self._seq, tag))
        # 只看 user 消息:system 提示词文案里也有 "<section_to_write>" 字面量,合并搜会错位
        user = messages[-1].get("content", "")
        if tag == "FINAL_REPORT_OUTLINE_JSON":
            return OUTLINE_JSON
        if tag == "FINAL_REPORT_MARKDOWN":
            title = _between(user, "<section_to_write>", "</section_to_write>").strip()
            return f"Body of {title}."
        if tag == "FINAL_REPORT_REVIEW_JSON":
            assert self.review is not None, "review call not expected in this test"
            if self.review == "GARBAGE":
                return "this is {{{ not json"
            return json.dumps(self.review)
        if tag == "REPAIR_JSON":
            return "still not json"  # 让 GARBAGE 场景的 repair 也失败
        if tag == "FINAL_REPORT_ADDITION_MARKDOWN":
            title = _between(user, "<section_to_write>", "</section_to_write>").strip()
            if title in self.addition_fail_titles:
                raise RuntimeError("addition boom")
            return f"Added body of {title}."
        raise AssertionError(f"unexpected tag: {tag}")


def _write(stub, monkeypatch, review_on):
    """review_on: True=显式开 / False=显式关(REPORT_REVIEW=0) / None=不设(默认,应为开)。"""
    monkeypatch.setenv("REPORT_PARALLEL", "1")
    if review_on is None:
        monkeypatch.delenv("REPORT_REVIEW", raising=False)
    else:
        monkeypatch.setenv("REPORT_REVIEW", "1" if review_on else "0")
    writer = FinalWriter(stub, "m")
    return asyncio.run(writer.write(
        original_question="Q?",
        summaries=["s1", "s2"],
        summary_sources=[["http://u1"], ["http://u2"]],
    ))


def _tags(stub):
    return [t for _, t in stub.calls]


# ---------- A1 关闸等价(默认开,关需显式 REPORT_REVIEW=0) ----------

def test_off_matches_baseline(monkeypatch):
    stub = ReviewStubClient(review=None)
    out = _write(stub, monkeypatch, review_on=False)
    assert out == BASELINE_REPORT
    assert "FINAL_REPORT_REVIEW_JSON" not in _tags(stub)


def test_default_unset_is_on(monkeypatch):
    """REPORT_REVIEW 不设 = 开(2026-07-20 终审拍板默认开)。"""
    stub = ReviewStubClient(review={"sections": [
        {"id": 0, "title": "Intro"}, {"id": 1}, {"id": 2}, {"id": 3},
    ]})
    out = _write(stub, monkeypatch, review_on=None)
    assert "FINAL_REPORT_REVIEW_JSON" in _tags(stub)
    assert out.startswith("## Intro\n\nAdded body of Intro.")


def test_summary_path_unchanged():
    stub = ReviewStubClient(review=None)
    out = asyncio.run(outline_then_parallel(
        FinalWriter(stub, "m"),
        question="Q?",
        source_pool=["src one", "src two"],
        sources_tag="findings",
        outline_system="outline sys",
        section_system="section sys",
        outline_tag="FINAL_REPORT_OUTLINE_JSON",
        section_tag="FINAL_REPORT_MARKDOWN",
        section_max_tokens=1000,
    ))
    assert out == BASELINE_BODY


# ---------- A2 重排+补节 ----------

def test_review_reorder_and_additions(monkeypatch):
    stub = ReviewStubClient(review={"sections": [
        {"id": 2},
        {"id": 0, "title": "Intro"},
        {"id": 1},
        {"id": 3},
        {"id": 0, "title": "Conclusion"},
    ]})
    out = _write(stub, monkeypatch, review_on=True)
    assert out == (
        "## Beta\n\nBody of Beta.\n\n"
        "## Intro\n\nAdded body of Intro.\n\n"
        "## Alpha\n\nBody of Alpha.\n\n"
        "## Gamma\n\nBody of Gamma.\n\n"
        "## Conclusion\n\nAdded body of Conclusion."
        "\n\n## References\n\n- http://u1\n- http://u2"
    )


# ---------- A3 审阅失败降级 ----------

def test_review_garbage_falls_back(monkeypatch):
    stub = ReviewStubClient(review="GARBAGE")
    out = _write(stub, monkeypatch, review_on=True)
    assert out == BASELINE_REPORT
    assert "FINAL_REPORT_REVIEW_JSON" in _tags(stub)
    assert "FINAL_REPORT_ADDITION_MARKDOWN" not in _tags(stub)


# ---------- A4 零内容丢失(丢节/重复/越界) ----------

def test_no_content_loss(monkeypatch):
    stub = ReviewStubClient(review={"sections": [
        {"id": 2}, {"id": 2}, {"id": 99},
    ]})
    out = _write(stub, monkeypatch, review_on=True)
    body = out.split("\n\n## References")[0]
    # 列过的 2 在前;漏列的 1、3 按原相对序补尾;重复只出现一次;越界 99 被忽略
    assert body == (
        "## Beta\n\nBody of Beta.\n\n"
        "## Alpha\n\nBody of Alpha.\n\n"
        "## Gamma\n\nBody of Gamma."
    )
    assert body.count("## Beta") == 1


# ---------- A5 补节失败不毁报 ----------

def test_addition_failure_skipped(monkeypatch):
    stub = ReviewStubClient(
        review={"sections": [
            {"id": 0, "title": "Intro"},
            {"id": 1}, {"id": 2}, {"id": 3},
            {"id": 0, "title": "Conclusion"},
        ]},
        addition_fail_titles={"Conclusion"},
    )
    out = _write(stub, monkeypatch, review_on=True)
    assert "## Intro\n\nAdded body of Intro." in out
    assert "Conclusion" not in out.split("\n\n## References")[0]
    for t in ("Alpha", "Beta", "Gamma"):
        assert f"Body of {t}." in out


# ---------- A6 上限 3 + 重名滤 ----------

def test_addition_cap_and_dedup(monkeypatch):
    stub = ReviewStubClient(review={"sections": [
        {"id": 1}, {"id": 2}, {"id": 3},
        {"id": 0, "title": "Alpha"},  # 与已有节重名 → 滤
        {"id": 0, "title": "N1"},
        {"id": 0, "title": "N2"},
        {"id": 0, "title": "N3"},
        {"id": 0, "title": "N4"},  # 超上限 → 弃
        {"id": 0, "title": "N5"},
    ]})
    out = _write(stub, monkeypatch, review_on=True)
    assert _tags(stub).count("FINAL_REPORT_ADDITION_MARKDOWN") == 3
    for t in ("N1", "N2", "N3"):
        assert f"## {t}\n\nAdded body of {t}." in out
    assert "## N4" not in out and "## N5" not in out
    assert out.count("## Alpha") == 1  # 重名新节没有第二个 Alpha


# ---------- A8 tag 与次序 ----------

def test_tags_and_ordering(monkeypatch):
    stub = ReviewStubClient(review={"sections": [
        {"id": 2}, {"id": 0, "title": "Intro"}, {"id": 1}, {"id": 3},
        {"id": 0, "title": "Conclusion"},
    ]})
    _write(stub, monkeypatch, review_on=True)
    seqs = {}
    for seq, tag in stub.calls:
        seqs.setdefault(tag, []).append(seq)
    assert len(seqs["FINAL_REPORT_REVIEW_JSON"]) == 1
    review_seq = seqs["FINAL_REPORT_REVIEW_JSON"][0]
    assert all(s < review_seq for s in seqs["FINAL_REPORT_MARKDOWN"])
    assert seqs["FINAL_REPORT_OUTLINE_JSON"][0] < min(seqs["FINAL_REPORT_MARKDOWN"])
    assert len(seqs["FINAL_REPORT_ADDITION_MARKDOWN"]) == 2
    assert all(s > review_seq for s in seqs["FINAL_REPORT_ADDITION_MARKDOWN"])
