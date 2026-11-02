"""实体消歧测试。

大模型实体抽取本身要 API key 才能跑，但「抽出来的名字怎么归并到标准 ID」是确定性逻辑，
不依赖模型，四层每一层都要能单独验证。
"""
from __future__ import annotations

from app.indexing.extract import Disambiguator, _char_bigram_sim, resolve_mentions


def _dis() -> Disambiguator:
    d = Disambiguator()
    # 固定用字面相似度，让测试结果不随 embedder 状态变化
    d._sim, d._threshold = _char_bigram_sim, Disambiguator.CHAR_SIM_THRESHOLD
    return d


def test_layer1_exact_name_and_alias():
    d = _dis()
    assert d.resolve("全勤奖") == ("config:full_attendance_bonus", "exact")
    assert d.resolve("VIP章节") == ("config:paid_chapter", "exact")
    assert d.resolve("作者收入") == ("effect:author_income", "exact")


def test_layer2_normalized():
    """全角字母、多余空格、标点都不该影响归并。"""
    d = _dis()
    eid, layer = d.resolve("ｆｕｌｌ_ａｔｔｅｎｄａｎｃｅ_ｂｏｎｕｓ")
    assert (eid, layer) == ("config:full_attendance_bonus", "normalized")
    eid, layer = d.resolve("首页 推荐位")
    assert (eid, layer) == ("config:home_banner", "normalized")


def test_layer3_similarity():
    d = _dis()
    eid, layer = d.resolve("新书期流量扶持计划")
    assert layer == "similarity"
    assert eid == "config:newbook_boost"


def test_layer4_unknown_goes_pending():
    """对不上的不能硬塞进最像的那个，要进待审核。"""
    d = _dis()
    assert d.resolve("月票奖励") == (None, "pending")


def test_aliases_are_tried_too():
    d = _dis()
    assert d.resolve("某个奇怪写法", aliases=["保底稿费"]) == ("config:guarantee_amount", "exact")


def test_resolve_mentions_groups_and_counts():
    mentions = [
        {"chunk_id": "c1", "name": "全勤奖", "type": "config", "aliases": []},
        {"chunk_id": "c2", "name": "全勤", "type": "config", "aliases": []},
        {"chunk_id": "c2", "name": "月票奖励", "type": "config", "aliases": []},
    ]
    found, pending, layers = resolve_mentions(mentions)
    assert found["config:full_attendance_bonus"] == ["c1", "c2"]
    assert [p["name"] for p in pending] == ["月票奖励"]
    assert layers["exact"] == 2 and layers["pending"] == 1
