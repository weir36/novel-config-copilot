"""知识图谱抽取质量评测：规则 vs LLM vs 混合。

gold 是 config_registry.yaml 里的 depends_on / requires_value / mutex_with /
enhances / weakens —— 注册表模拟的是配置平台元数据库，它知道正确答案；
而抽取器只能看 raw_docs 里的散文。这正好构成一个干净的评测：
**能不能只靠人写的文档，把结构化关系还原出来。**

结构性关系（BELONGS_TO / APPLIES_TO / CONSTRAINED_BY）不参与评测，
它们本来就是从元数据直接读的，不需要抽取，算进去会虚高。

这个评测回答的是一个具体的选型问题：这一步值不值得花 LLM 的钱。
如果规则抽取的 F1 已经接近 LLM，那就用规则（快、免费、可解释）；
如果 LLM 在某几类关系上显著更好，那就只在那几类上用 LLM。

另外还有一组**实体评测**（--entities）：对比词表和「大模型抽取 + 四层消歧」两种实体抽法。
参照答案是每个 chunk 里词表能认出的配置项和业务效果 —— 所以词表自己按定义就是 1.0，
这一组真正要看的是：大模型抽取 + 消歧之后，能还原出多少、有多少被归错、有多少进了待审核。

用法：
    python -m eval.run_graph_eval --extractors rule           # 离线可跑
    python -m eval.run_graph_eval --extractors rule,llm,hybrid # 需要 LLM_API_KEY
    python -m eval.run_graph_eval --entities vocab,llm         # 实体评测，llm 需要 LLM_API_KEY
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from app.foundation.registry import get_registry
from app.foundation.settings import settings
from app.foundation.models import KnowledgeGraph, RelationType
from app.indexing.chunker import chunk_all
from app.generation.llm_client import llm_meta
from app.indexing.extract import (
    LLMEntityExtractor,
    LLMExtractor,
    RuleExtractor,
    merge_relations,
    resolve_mentions,
)

OUT_PATH = settings.root_dir / "eval" / "results" / "graph_eval.json"

# 只评测「需要从散文里抽」的关系类型
EVAL_TYPES = [
    RelationType.DEPENDS_ON,
    RelationType.REQUIRES_VALUE,
    RelationType.MUTEX_WITH,
    RelationType.ENHANCES,
    RelationType.WEAKENS,
]

Edge = tuple[str, str, str]  # (source_id, target_id, type)


def normalize(edge: Edge) -> Edge:
    """互斥无序归一，其余保持方向。"""
    src, dst, rtype = edge
    if rtype == RelationType.MUTEX_WITH.value:
        src, dst = sorted([src, dst])
    return (src, dst, rtype)


def gold_edges() -> set[Edge]:
    reg = get_registry()
    out: set[Edge] = set()
    for key, item in reg.items.items():
        src = f"config:{key}"
        for dep in item.depends_on:
            out.add(normalize((src, f"config:{dep}", RelationType.DEPENDS_ON.value)))
        for dep in item.requires_value:
            out.add(normalize((src, f"config:{dep}", RelationType.REQUIRES_VALUE.value)))
        for other in item.mutex_with:
            out.add(normalize((src, f"config:{other}", RelationType.MUTEX_WITH.value)))
        for e in item.enhances:
            out.add(normalize((src, f"effect:{e}", RelationType.ENHANCES.value)))
        for e in item.weakens:
            out.add(normalize((src, f"effect:{e}", RelationType.WEAKENS.value)))
    return out


def pred_edges(relations) -> set[Edge]:
    return {
        normalize((r.source_id, r.target_id, r.type.value))
        for r in relations
        if r.type in EVAL_TYPES
    }


def prf(pred: set[Edge], gold: set[Edge]) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "pred": len(pred), "gold": len(gold)}


def by_type(pred: set[Edge], gold: set[Edge]) -> dict:
    out = {}
    for t in EVAL_TYPES:
        p = {e for e in pred if e[2] == t.value}
        g = {e for e in gold if e[2] == t.value}
        if g:
            out[t.value] = prf(p, g)
    return out


def run(extractor: str, chunks) -> set[Edge]:
    if extractor == "rule":
        return pred_edges(RuleExtractor().extract(chunks))
    if extractor == "llm":
        return pred_edges(LLMExtractor().extract(chunks))
    rule_rels = RuleExtractor().extract(chunks)
    llm_rels = LLMExtractor().extract(chunks)
    return pred_edges(merge_relations(rule_rels, llm_rels))


def entity_reference(chunks) -> set[tuple[str, str]]:
    """参照答案：(chunk_id, 实体 id)。配置项用 chunk 上的 linked_item_keys，业务效果按名字匹配。"""
    reg = get_registry()
    out: set[tuple[str, str]] = set()
    for c in chunks:
        for key in c.linked_item_keys:
            out.add((c.chunk_id, f"config:{key}"))
        for key, name in reg.effects.items():
            if name in c.text:
                out.add((c.chunk_id, f"effect:{key}"))
    return out


def run_entities(mode: str, chunks) -> tuple[set[tuple[str, str]], dict]:
    if mode == "vocab":
        return entity_reference(chunks), {"pending": 0, "layers": {}}
    mentions = LLMEntityExtractor().extract(chunks)
    found, pending, layers = resolve_mentions(mentions)
    pairs = {
        (cid, eid) for eid, cids in found.items() for cid in cids
        if eid.startswith(("config:", "effect:"))
    }
    return pairs, {"pending": len(pending), "pending_names": [p["name"] for p in pending][:30],
                   "layers": layers, "mentions": len(mentions)}


def entity_eval(modes: list[str], chunks) -> dict:
    ref = entity_reference(chunks)
    print(f"\n实体评测：参照答案 {len(ref)} 个（chunk, 实体）对\n")
    if "llm" in modes and llm_meta()["provider"] == "mock":
        print("  ⚠️  当前为 MockLLM（未配置 LLM_API_KEY），llm 实体抽取不会产出任何结果，已跳过。\n")
        modes = [m for m in modes if m != "llm"]
    out = {}
    for mode in modes:
        pred, info = run_entities(mode, chunks)
        res = {**prf(pred, ref), **info}
        out[mode] = res
        extra = (f"  消歧层命中 {info['layers']}  待审核 {info['pending']} 个"
                 if mode == "llm" else "  （参照答案就是词表，按定义为 1.0）")
        print(f"  {mode:6s} P={res['precision']:.3f}  R={res['recall']:.3f}  F1={res['f1']:.3f}{extra}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractors", default="rule")
    ap.add_argument("--entities", default="", help="实体评测，如 vocab,llm；不传则跳过")
    ap.add_argument("--verbose", action="store_true", help="打印漏抽/误抽明细")
    args = ap.parse_args()

    chunks = chunk_all()
    gold = gold_edges()
    print(f"图谱抽取评测：{len(chunks)} 个 chunk，gold 关系 {len(gold)} 条\n")

    results = {}
    preds: dict[str, set[Edge]] = {}
    for ext in [e.strip() for e in args.extractors.split(",") if e.strip()]:
        pred = run(ext, chunks)
        preds[ext] = pred
        overall = prf(pred, gold)
        results[ext] = {"overall": overall, "by_type": by_type(pred, gold)}
        print(f"  {ext:8s} P={overall['precision']:.3f}  R={overall['recall']:.3f}  "
              f"F1={overall['f1']:.3f}  (抽出 {overall['pred']} 条，命中 {overall['tp']} 条)")

    print("\n  分关系类型 F1：")
    types = [t.value for t in EVAL_TYPES]
    print("  " + "extractor".ljust(11) + "".join(t[:12].ljust(15) for t in types))
    for ext, res in results.items():
        row = "  " + ext.ljust(11)
        for t in types:
            v = res["by_type"].get(t)
            row += (f"{v['f1']:.3f}" if v else "-").ljust(15)
        print(row)

    if args.verbose:
        for ext, pred in preds.items():
            reg = get_registry()

            def label(eid: str) -> str:
                key = eid.split(":", 1)[1]
                return reg.items[key].name if key in reg.items else reg.effects.get(key, key)

            print(f"\n  [{ext}] 漏抽 {len(gold - pred)} 条：")
            for e in sorted(gold - pred)[:25]:
                print(f"    {label(e[0])} -[{e[2]}]-> {label(e[1])}")
            print(f"  [{ext}] 误抽 {len(pred - gold)} 条：")
            for e in sorted(pred - gold)[:25]:
                print(f"    {label(e[0])} -[{e[2]}]-> {label(e[1])}")

    entity_results = {}
    if args.entities:
        entity_results = entity_eval([m.strip() for m in args.entities.split(",") if m.strip()], chunks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"n_chunks": len(chunks), "n_gold": len(gold), "results": results,
                    "entities": entity_results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {OUT_PATH.relative_to(settings.root_dir)}")


if __name__ == "__main__":
    main()
