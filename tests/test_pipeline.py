"""检索、方案抽取、追问识别与在线链路的冒烟测试。

这些测试不验证「回答得好不好」（那是评测脚本的事），只验证
**链路不会崩、契约不会变、降级路径真的能走通**。
"""
from __future__ import annotations

import pytest

from app.validation.plan_extract import extract_plan
from app.agent.tools import run_tool
from app.chat.pipeline import handle_chat
from app.foundation.settings import settings
from app.intent.classifier import get_classifier
from app.retrieval.graphrag import get_retriever
from app.foundation.models import ChatRequest, Intent

pytestmark = pytest.mark.skipif(
    not (settings.index_dir / "chunks_vectors.npy").exists(),
    reason="索引未构建，先执行 python -m app.indexing.build --extractor rule",
)


# ---------------------------------------------------------------- 检索
def test_entity_linking_exact_key():
    """编辑直接贴配置项 key 时必须精确命中，不能被向量召回稀释。"""
    seeds = get_retriever().linker.link("full_attendance_bonus 怎么配")
    assert any(s.entity.props.get("key") == "full_attendance_bonus" for s in seeds)


def test_entity_linking_alias():
    """别名也要能链接到标准实体。"""
    seeds = get_retriever().linker.link("千字保底给多少合适")
    keys = {s.entity.props.get("key") for s in seeds}
    assert "guarantee_amount" in keys


def test_mentions_only_returns_literal_hits():
    """mentions 只认原样出现的实体，不做向量召回 —— 没提配置项就该是空的。"""
    linker = get_retriever().linker
    assert "config:full_attendance_bonus" in linker.mentions("全勤奖是什么")
    assert linker.mentions("上限是多少") == []


def test_graph_expansion_finds_multi_hop_dependency():
    """加更活动的 2 跳依赖链（→全勤奖→日更字数要求）必须出现在子图里。

    这是 GraphRAG 的核心能力，挂了就说明图或扩展策略坏了。
    """
    res = get_retriever().retrieve("加更活动怎么开", mode="graphrag")
    keys = {e.entity.props.get("key") for e in res.subgraph_entities}
    assert "full_attendance_bonus" in keys, "1 跳依赖未扩展出来"
    assert "daily_update_words" in keys, "2 跳依赖未扩展出来"


def test_triples_are_generated():
    res = get_retriever().retrieve("免费阅读模式和付费章节", mode="graphrag")
    assert res.triples
    assert any("互斥" in t.to_text() for t in res.triples)


def test_context_text_contains_both_parts():
    res = get_retriever().retrieve("首页推荐位依赖什么", mode="graphrag")
    ctx = res.context_text()
    assert "知识图谱关系" in ctx
    assert "领域知识片段" in ctx


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid", "graphrag"])
def test_all_retrieval_modes_return_chunks(mode):
    res = get_retriever().retrieve("作者分成比例的取值范围", mode=mode)
    assert res.chunks, f"{mode} 模式未召回任何文本"


def test_graphrag_falls_back_when_no_entity_linked():
    """实体链接落空时应退化为 hybrid，而不是返回空。"""
    res = get_retriever().retrieve("阿巴阿巴阿巴", mode="graphrag")
    assert res.chunks


# ---------------------------------------------------------------- 意图
@pytest.mark.parametrize("query,expected", [
    ("全勤奖是什么意思", Intent.EXPLAIN),
    ("作者问为什么这个月没拿到全勤", Intent.EXPLAIN),
    ("给一本新书配一套冷启动方案", Intent.RECOMMEND),
    ("在当前配置基础上帮我提高作者收入", Intent.OPTIMIZE),
    ("今天天气怎么样", Intent.OUT_OF_SCOPE),
])
def test_intent_classification(query, expected):
    assert get_classifier().classify(query).intent == expected


# ---------------------------------------------------------------- 方案抽取
def test_extract_plan_from_markdown_table():
    answer = """
## 配置清单

| 配置项 | 取值 | 维度 | 理由 |
|--------|------|------|------|
| 新人作者扶持计划（new_author_plan） | 开启 | 作者 | 新人标配 |
| 日更字数要求（daily_update_words） | 4000 字 | 作品 | 全勤考核线 |
| 签约类型（contract_type） | 保底分成 | 作品 | 作者担心前期收入 |
| 作者分成比例（share_ratio） | 45% | 作品 | 保底作品不超过 50% |
"""
    plan = extract_plan(answer, use_llm_fallback=False)
    assert plan.values["new_author_plan"] is True
    assert plan.values["daily_update_words"] == 4000
    assert plan.values["contract_type"] == "guarantee"
    assert plan.values["share_ratio"] == 45
    assert plan.dimensions["new_author_plan"] == "author"
    assert plan.dimensions["daily_update_words"] == "book"


def test_extract_plan_value_with_annotation():
    """真实回答里取值常带括号注释，早先整行被跳过，合法方案被当成「没抽到方案」。"""
    answer = (
        "| 配置项 | 当前值 | 建议值 | 理由 |\n| :--- | :--- | :--- | :--- |\n"
        "| 首页推荐位（home_banner） | False（未配置） | True（作品维度） | 曝光 |\n"
        "| 日更字数要求（daily_update_words） | 4000 | 6000（字/日） | 冲榜 |\n"
    )
    plan = extract_plan(answer, use_llm_fallback=False)
    assert plan.values == {"home_banner": True, "daily_update_words": 6000}
    assert plan.dimensions.get("home_banner") == "book"


def test_extract_plan_from_change_table():
    """变更项表格是「配置项 | 当前值 | 建议值 | 理由」，取值应读第三列。"""
    answer = """
| 配置项 | 当前值 | 建议值 | 理由 |
|--------|--------|--------|------|
| 付费起始章节（paywall_start_chapter） | 15 | 30 | 前期先留住读者 |
"""
    plan = extract_plan(answer, use_llm_fallback=False)
    assert plan.values["paywall_start_chapter"] == 30


def test_extract_plan_ignores_unknown_items():
    answer = "| 配置项 | 取值 |\n|---|---|\n| 不存在的配置项 | 开启 |\n"
    assert not extract_plan(answer, use_llm_fallback=False).values


# ---------------------------------------------------------------- 工具
def test_tool_get_config_detail_includes_dependency_chain():
    res = run_tool("get_config_detail", {"keys": ["extra_update_event"]})
    assert res.ok
    assert "依赖链" in res.content
    assert "全勤奖" in res.content


def test_tool_check_constraints_merges_current_values():
    """base_scope 必须把当前配置合并进来 —— 否则「新配项 × 老配项」的冲突发现不了。

    B004 是买断作品，单独提交「开完本奖」看似无害，合并后应被 R001 拦下。
    """
    res = run_tool("check_constraints", {
        "values": {"completion_bonus": True},
        "base_scope": "book:B004",
    })
    assert res.ok
    assert res.data["passed"] is False
    assert any(v["rule_id"] == "R001" for v in res.data["errors"])


def test_tool_get_current_config_merges_inheritance():
    """作品的生效配置应按 频道 < 作者 < 作品 合并。"""
    res = run_tool("get_current_config", {"scope": "book", "id": "B001"})
    assert res.ok
    assert res.data["origin"]["contract_type"] == "book"
    assert res.data["origin"]["new_author_plan"] == "author"
    assert res.data["origin"]["newbook_boost"] == "channel"


def test_tool_find_items_by_effect_accepts_chinese():
    res = run_tool("find_items_by_effect", {"effect": "新书曝光"})
    assert res.ok
    assert "home_banner" in res.data["enhances"]


def test_tool_unknown_name_returns_readable_error():
    """工具不存在时要返回可读反馈供 Agent 自纠，而不是抛异常。"""
    res = run_tool("no_such_tool", {})
    assert not res.ok
    assert "不存在" in res.content


def test_tool_bad_arguments_do_not_raise():
    res = run_tool("get_current_config", {"wrong_arg": 1})
    assert not res.ok
    assert "参数错误" in res.content


# ---------------------------------------------------------------- 追问识别
def _chat(sid: str, message: str):
    return handle_chat(ChatRequest(session_id=sid, message=message, use_agent=False))


def _fresh(sid: str) -> str:
    from app.session.session import get_session_manager  # noqa: PLC0415

    get_session_manager().reset(sid)
    return sid


def test_first_turn_is_never_followup():
    resp = _chat(_fresh("t-fu-first"), "全勤奖是什么")
    assert resp.followup is not None
    assert resp.followup.is_followup is False


def test_pronoun_followup():
    sid = _fresh("t-fu-pron")
    _chat(sid, "全勤奖是什么")
    resp = _chat(sid, "那加更活动呢")
    assert resp.followup.is_followup
    assert "指代或承接词" in resp.followup.reasons


def test_graph_neighbor_followup():
    """问完首页推荐位接着问精品审核：两者在图上直接相连，算追问。"""
    sid = _fresh("t-fu-nbr")
    _chat(sid, "首页推荐位依赖什么")
    resp = _chat(sid, "精品审核怎么过")
    assert resp.followup.is_followup
    assert "配置项和上一轮重合" in resp.followup.reasons


def test_new_topic_is_not_followup_and_gets_no_history():
    """新话题不带历史、不做检索剪枝。"""
    sid = _fresh("t-fu-new")
    _chat(sid, "首页推荐位依赖什么")
    resp = _chat(sid, "保底分成的作者分成比例最多多少")
    assert resp.followup.is_followup is False
    assert resp.retrieval is not None
    assert not resp.retrieval.pruned_entity_ids


# ---------------------------------------------------------------- 在线链路
def test_out_of_scope_short_circuits():
    """越界问题不得进入检索和 Agent —— 这是成本控制的关键路径。"""
    resp = handle_chat(ChatRequest(session_id="t-oos", message="今天北京天气怎么样"))
    assert resp.intent.intent == Intent.OUT_OF_SCOPE
    assert resp.retrieval is None
    assert resp.agent_trace is None


def test_chat_explain_path_runs():
    resp = _chat("t-explain", "全勤奖是什么")
    assert resp.answer
    assert resp.retrieval is not None
    assert resp.retrieval.chunks


def test_session_followup_prunes_seen_entities():
    """追问时，上一轮扩展过的实体应进入剪枝集合；本轮种子不能被剪。"""
    from app.session.session import get_session_manager  # noqa: PLC0415

    sid = _fresh("t-multi")
    sm = get_session_manager()
    _chat(sid, "加更活动是什么")
    state = sm.load(sid)
    assert state.seen_entity_ids

    pruned = sm.prune_entity_ids(state, ["config:extra_update_event"])
    assert "config:extra_update_event" not in pruned
    assert pruned
