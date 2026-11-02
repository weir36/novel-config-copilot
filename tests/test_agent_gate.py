"""收尾闸门：方案先合并书的当前配置再校验；不合规就退回 Agent 重改。用脚本化的 LLM，不调真实接口。"""
from __future__ import annotations

import json

import pytest

import app.agent.loop as loop
from app.agent.tools import detect_scope, validate_with_base
from app.foundation.models import ConfigPlan, Intent


def test_detect_scope_prefers_book():
    assert detect_scope("给 W1001 的新书 B001 配一套方案") == "book:B001"
    assert detect_scope("W2002 这个作者") == "author:W2002"
    assert detect_scope("全勤奖是什么") is None


def test_change_set_is_validated_against_current_config():
    # B004 已开付费章节：只改付费起点是合法的，单看变更项会误判成缺依赖
    change = ConfigPlan(values={"paywall_start_chapter": 30})
    assert not validate_with_base(change, None)[0].passed
    assert validate_with_base(change, "book:B004")[0].passed
    # B004 是买断：新开完本奖要被 R001 拦下（新配项 × 老配项）
    bad = ConfigPlan(values={"paywall_start_chapter": 30, "completion_bonus": True})
    report, _ = validate_with_base(bad, "book:B004")
    assert [v.rule_id for v in report.errors] == ["R001"]


def _answer(rows: str) -> str:
    return f"## 变更项\n| 配置项 | 当前值 | 建议值 | 理由 |\n|---|---|---|---|\n{rows}"


class ScriptedLLM:
    """按顺序返回预设的 Agent 决策。"""

    def __init__(self, decisions: list[dict]):
        self.decisions = decisions
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, messages, **_kw):
        self.prompts.append(messages[-1]["content"])
        decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return decision


@pytest.fixture
def scripted(monkeypatch):
    def install(decisions):
        llm = ScriptedLLM(decisions)
        monkeypatch.setattr(loop, "chat_json", llm)
        return llm
    return install


def _final(rows: str) -> dict:
    return {"thought": "给出方案", "action": "final_answer", "action_input": {"answer": _answer(rows)}}


def test_violating_plan_is_sent_back_and_repaired(scripted):
    llm = scripted([
        _final("| completion_bonus | false | true | 冲完本 |"),           # 买断书开完本奖，违反 R001
        _final("| paywall_start_chapter | 15 | 30 | 提升前期留存 |"),  # 改正后合法
    ])
    trace = loop.ConfigAgent(max_steps=4).run("把 B004 的付费起点改到第 30 章，再开个完本奖", Intent.OPTIMIZE)
    assert llm.calls == 2
    assert "R001" in llm.prompts[1]  # 违规项被原样退回
    assert "paywall_start_chapter" in trace.final_answer and "completion_bonus" not in trace.final_answer
    assert trace.stopped_reason == "final"


def test_repairs_are_capped(scripted):
    llm = scripted([_final("| completion_bonus | false | true | 冲完本 |")])  # 怎么退回都不改
    trace = loop.ConfigAgent(max_steps=4).run("B004 开个完本奖", Intent.OPTIMIZE)
    assert llm.calls == loop.MAX_REPAIRS + 1
    assert trace.stopped_reason == "final"  # 交给 pipeline 标「未通过，请勿直接使用」


def test_legal_change_set_passes_first_time(scripted):
    llm = scripted([_final("| paywall_start_chapter | 15 | 30 | 提升前期留存 |")])
    loop.ConfigAgent(max_steps=4).run("把 B004 的付费起点改到第 30 章", Intent.OPTIMIZE)
    assert llm.calls == 1


def test_pipeline_marks_unrepaired_plan_as_blocked():
    from app.chat.pipeline import _validation_suffix

    report, _ = validate_with_base(ConfigPlan(values={"completion_bonus": True}), "book:B004")
    assert "请勿直接使用" in _validation_suffix(report)
    json.dumps(report.model_dump(mode="json"))  # 流式接口要能序列化
