"""规则引擎测试。

这是系统里唯一一个「必须 100% 正确」的组件 —— 检索差几个点只是体验问题，
规则判错就是把违规配置放到线上。所以这里的用例直接对着 R001~R004
和每一类依赖/互斥场景逐条写。
"""
from __future__ import annotations

import pytest

from app.foundation.registry import get_registry
from app.validation.rule_engine import get_rule_engine
from app.foundation.models import ConfigPlan, ViolationSeverity


@pytest.fixture
def engine():
    return get_rule_engine()


def ids(report) -> set[str]:
    return {v.rule_id for v in report.violations}


# ---------------------------------------------------------------- 依赖
def test_dependency_chain_missing(engine):
    """只开加更活动，上游的全勤奖和日更字数都没配 —— 应该报依赖缺失。"""
    report = engine.validate(ConfigPlan(values={"extra_update_event": True}))
    assert not report.passed
    assert "DEPENDENCY_MISSING" in ids(report)


def test_dependency_chain_satisfied(engine):
    report = engine.validate(ConfigPlan(values={
        "daily_update_words": 4000, "full_attendance_bonus": True, "extra_update_event": True,
    }))
    assert "DEPENDENCY_MISSING" not in ids(report)
    assert report.passed


def test_disabled_item_does_not_trigger_dependency(engine):
    """关着的配置项不应触发依赖校验 —— 否则编辑关掉一个开关反而会报错。"""
    report = engine.validate(ConfigPlan(values={"extra_update_event": False}))
    assert "DEPENDENCY_MISSING" not in ids(report)


@pytest.mark.parametrize("item,dep", [
    ("home_banner", "quality_review_pass"),
    ("ad_revenue_share", "free_read_mode"),
    ("paywall_start_chapter", "paid_chapter"),
])
def test_single_dependencies(engine, item, dep):
    value = 30 if item == "paywall_start_chapter" else True
    bad = engine.validate(ConfigPlan(values={item: value}))
    assert "DEPENDENCY_MISSING" in ids(bad)
    good = engine.validate(ConfigPlan(values={item: value, dep: True}))
    assert "DEPENDENCY_MISSING" not in ids(good)


def test_requires_value(engine):
    """保底金额要求签约类型 = 保底分成。"""
    bad = engine.validate(ConfigPlan(values={"guarantee_amount": 20, "contract_type": "share"}))
    assert "REQUIRED_VALUE_UNMET" in ids(bad)

    good = engine.validate(ConfigPlan(values={"guarantee_amount": 20, "contract_type": "guarantee"}))
    assert "REQUIRED_VALUE_UNMET" not in ids(good)


# ---------------------------------------------------------------- 互斥
@pytest.mark.parametrize("a,b", [
    ("new_author_plan", "veteran_author_plan"),
    ("free_read_mode", "paid_chapter"),
])
def test_mutex_pairs(engine, a, b):
    report = engine.validate(ConfigPlan(values={a: True, b: True}))
    assert "MUTEX_CONFLICT" in ids(report)


def test_mutex_is_symmetric_and_deduped(engine):
    """互斥双向各报一次会让编辑看到重复错误，应该去重成一条。"""
    report = engine.validate(ConfigPlan(values={"free_read_mode": True, "paid_chapter": True}))
    mutex = [v for v in report.violations if v.rule_id == "MUTEX_CONFLICT"]
    assert len(mutex) == 1


# ---------------------------------------------------------------- 组合规则
def test_r001_buyout_no_completion_bonus(engine):
    bad = engine.validate(ConfigPlan(values={"contract_type": "buyout", "completion_bonus": True}))
    assert "R001" in ids(bad)
    assert not bad.passed

    ok = engine.validate(ConfigPlan(values={"contract_type": "guarantee", "completion_bonus": True,
                                            "guarantee_amount": 20}))
    assert "R001" not in ids(ok)


def test_r002_attendance_needs_4000_words(engine):
    bad = engine.validate(ConfigPlan(values={"full_attendance_bonus": True, "daily_update_words": 2000}))
    assert "R002" in ids(bad)

    ok = engine.validate(ConfigPlan(values={"full_attendance_bonus": True, "daily_update_words": 4000}))
    assert "R002" not in ids(ok)  # 4000 刚好压线


def test_r002_zero_words_is_dependency_not_r002(engine):
    """日更字数为 0 时是依赖缺失，不该再重复报 R002。"""
    report = engine.validate(ConfigPlan(values={"full_attendance_bonus": True, "daily_update_words": 0}))
    assert "DEPENDENCY_MISSING" in ids(report)
    assert "R002" not in ids(report)


def test_r003_guarantee_share_cap(engine):
    bad = engine.validate(ConfigPlan(values={
        "contract_type": "guarantee", "guarantee_amount": 20, "share_ratio": 60}))
    assert "R003" in ids(bad)

    ok = engine.validate(ConfigPlan(values={
        "contract_type": "guarantee", "guarantee_amount": 20, "share_ratio": 50}))
    assert "R003" not in ids(ok)

    # 普通分成作品给 60% 没问题
    share = engine.validate(ConfigPlan(values={"contract_type": "share", "share_ratio": 60}))
    assert "R003" not in ids(share)


def test_r004_is_warn_not_error(engine):
    """付费起点过早是 warn 级：可以提交，但必须提示。"""
    report = engine.validate(ConfigPlan(values={"paid_chapter": True, "paywall_start_chapter": 15}))
    r004 = [v for v in report.violations if v.rule_id == "R004"]
    assert r004 and r004[0].severity == ViolationSeverity.WARN
    assert report.passed  # warn 不阻断

    later = engine.validate(ConfigPlan(values={"paid_chapter": True, "paywall_start_chapter": 25}))
    assert "R004" not in ids(later)


# ---------------------------------------------------------------- 默认值参与校验
def test_defaults_participate_in_composite_rules(engine):
    """只提交一个字段时，组合规则要拿注册表默认值补齐另一边。

    只配保底分成，作者分成走默认值 50% —— 刚好不超；
    再把默认值换成显式 55%，就应当被 R003 拦下。
    """
    report = engine.validate(ConfigPlan(values={"contract_type": "guarantee"}))
    assert "R003" not in ids(report)
    report = engine.validate(ConfigPlan(values={"contract_type": "guarantee", "share_ratio": 55}))
    assert "R003" in ids(report)


# ---------------------------------------------------------------- 取值与维度
def test_invalid_enum_value(engine):
    report = engine.validate(ConfigPlan(values={"contract_type": "lease"}))
    assert "INVALID_VALUE" in ids(report)


def test_out_of_range_value(engine):
    report = engine.validate(ConfigPlan(values={"share_ratio": 80}))
    assert "INVALID_VALUE" in ids(report)


def test_unknown_item(engine):
    report = engine.validate(ConfigPlan(values={"not_a_real_config": True}))
    assert "UNKNOWN_ITEM" in ids(report)


def test_invalid_dimension(engine):
    """签约类型只能配在作品维度，配到频道维度应被拦。"""
    report = engine.validate(ConfigPlan(
        values={"contract_type": "share"},
        dimensions={"contract_type": "channel"},
    ))
    assert "INVALID_DIMENSION" in ids(report)


# ---------------------------------------------------------------- 自动修复
def test_auto_fix_completes_dependency_chain(engine):
    plan = ConfigPlan(values={"extra_update_event": True})
    fixed, notes = engine.auto_fix(plan)
    assert fixed.values.get("full_attendance_bonus") is True
    assert fixed.values.get("daily_update_words", 0) > 0
    assert notes


def test_auto_fix_does_not_resolve_business_tradeoffs(engine):
    """互斥冲突涉及业务取舍，系统不该替编辑决定关哪一个。"""
    plan = ConfigPlan(values={"free_read_mode": True, "paid_chapter": True})
    fixed, _notes = engine.auto_fix(plan)
    assert fixed.values["free_read_mode"] is True
    assert fixed.values["paid_chapter"] is True
    assert not engine.validate(fixed).passed


# ---------------------------------------------------------------- 注册表
def test_dependency_chain_has_no_cycle():
    reg = get_registry()
    for key in reg.items:
        chain = reg.dependency_chain(key)
        assert key not in chain, f"{key} 的依赖链成环"


def test_mutex_is_declared_both_ways():
    reg = get_registry()
    for key, item in reg.items.items():
        for other in item.mutex_with:
            assert key in reg.items[other].mutex_with, f"{key} 与 {other} 的互斥只写了一边"


@pytest.mark.parametrize("values", [
    # 方案一：新人作者的新书冷启动
    {"new_author_plan": True, "newbook_boost": True, "free_read_mode": True,
     "ad_revenue_share": True, "daily_update_words": 4000, "full_attendance_bonus": True},
    # 方案二：头部作品冲榜
    {"quality_review_pass": True, "home_banner": True, "extra_update_event": True,
     "full_attendance_bonus": True, "daily_update_words": 6000},
])
def test_all_case_documents_are_consistent(engine, values):
    """《编辑操作指南与典型方案》里的方案必须真的能通过校验 —— 否则就是文档在教人配错。"""
    report = engine.validate(ConfigPlan(values=values))
    assert report.passed, report.to_text()
