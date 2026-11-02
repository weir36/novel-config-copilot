"""检索消融评测：GraphRAG 到底带来了多少增益。

评测的是 **item 级召回**，不是 chunk 级。原因是 chunk_id 会随切块策略变化，
换个切块方案指标就不可比了；而"回答这个问题需要哪些配置项的知识"是稳定的。

对每条 query：
    retrieved_items = 召回文本段上的 linked_item_keys 并集（+ GraphRAG 模式下的子图实体）
    recall    = |retrieved ∩ gold| / |gold|
    precision = |retrieved ∩ gold| / |retrieved|
另外报 hit@k（gold 是否全部命中）和按 hop 分层的召回 —— 后者用来回答
"增益到底来自哪"：如果 GraphRAG 的提升集中在 hop=2 的问题上，
说明确实是多跳关系在起作用，而不是碰巧调参调出来的。

用法：
    python -m eval.run_retrieval_eval
    python -m eval.run_retrieval_eval --modes vector,hybrid,graphrag --top-k 6
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.foundation.settings import settings
from app.retrieval.graphrag import get_retriever

EVAL_PATH = settings.gold_dir / "retrieval_eval.jsonl"
OUT_PATH = settings.root_dir / "eval" / "results" / "retrieval_eval.json"

ALL_MODES = ["vector", "bm25", "hybrid", "graphrag"]


def load_cases() -> list[dict]:
    cases = []
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_comment" in obj:
            continue
        cases.append(obj)
    return cases


def retrieved_item_keys(result, include_subgraph: bool, strict: bool) -> set[str]:
    """两种口径。

    宽口径（linked）：chunk 里提到即算命中。
    严格口径（primary）：只有「小节标题就是讲这个配置项」的 chunk 才算命中。

    两个都要报。宽口径在小语料上会虚高到 0.93+（每个 chunk 平均提到 2.65 个配置项，
    top6 期望覆盖 15.9 个，而 gold 平均只有 4 个 —— 等于送分），
    单看它会得出"GraphRAG 没用"的错误结论。严格口径才能区分开
    "召回到了讲这个配置项的那一段" 和 "碰巧有段文字提了它一嘴"。
    """
    keys: set[str] = set()
    for c in result.chunks:
        keys.update(c.chunk.primary_item_keys if strict else c.chunk.linked_item_keys)
    if include_subgraph:
        # GraphRAG 模式下子图实体本身也是"召回到的知识"：
        # 实体的三元组（依赖/互斥/效果）会进上下文，LLM 能直接用。
        # 不算进去会低估图谱的贡献 —— 图谱的价值恰恰是补上文本没召回到的那部分结构。
        for e in result.subgraph_entities:
            k = e.entity.props.get("key")
            if k and e.entity.type.value == "ConfigItem":
                keys.add(k)
    return keys


def eval_mode(mode: str, cases: list[dict], top_k: int, strict: bool = True) -> dict:
    retriever = get_retriever()
    recalls: list[float] = []
    precisions: list[float] = []
    loose_recalls: list[float] = []
    full_hits = 0
    latencies: list[float] = []
    by_hop: dict[int, list[float]] = {}
    per_case: list[dict] = []

    for case in cases:
        gold = set(case.get("gold_items", []))
        t0 = time.perf_counter()
        result = retriever.retrieve(case["query"], mode=mode, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        include_sub = mode == "graphrag"
        got = retrieved_item_keys(result, include_subgraph=include_sub, strict=strict)
        got_loose = retrieved_item_keys(result, include_subgraph=include_sub, strict=False)
        if gold:
            loose_recalls.append(len(gold & got_loose) / len(gold))
        if not gold:
            # 无 gold_items 的用例（如维度优先级类问题）只看是否召回到指定文档
            gold_docs = set(case.get("gold_docs", []))
            hit = any(c.chunk.doc_id in gold_docs for c in result.chunks) if gold_docs else True
            recalls.append(1.0 if hit else 0.0)
            precisions.append(1.0 if hit else 0.0)
            full_hits += int(hit)
            per_case.append({"id": case["id"], "recall": 1.0 if hit else 0.0, "gold": [],
                             "missed": []})
            continue

        inter = gold & got
        recall = len(inter) / len(gold)
        precision = len(inter) / len(got) if got else 0.0
        recalls.append(recall)
        precisions.append(precision)
        full_hits += int(recall >= 0.999)
        by_hop.setdefault(int(case.get("hop", 1)), []).append(recall)
        per_case.append({
            "id": case["id"], "query": case["query"], "recall": round(recall, 3),
            "precision": round(precision, 3), "hop": case.get("hop", 1),
            "gold": sorted(gold), "missed": sorted(gold - got),
        })

    mean_r = statistics.mean(recalls)
    mean_p = statistics.mean(precisions)
    f1 = 2 * mean_r * mean_p / (mean_r + mean_p) if (mean_r + mean_p) else 0.0
    return {
        "mode": mode,
        "recall": round(mean_r, 4),
        "recall_loose": round(statistics.mean(loose_recalls), 4) if loose_recalls else 0.0,
        "precision": round(mean_p, 4),
        "f1": round(f1, 4),
        "full_hit_rate": round(full_hits / len(cases), 4),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        "recall_by_hop": {
            str(h): round(statistics.mean(v), 4) for h, v in sorted(by_hop.items())
        },
        "per_case": per_case,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--top-k", type=int, default=settings.final_top_k)
    ap.add_argument("--verbose", action="store_true", help="打印每条用例的漏召回明细")
    args = ap.parse_args()

    cases = load_cases()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"检索消融评测：{len(cases)} 条用例，top_k={args.top_k}\n")

    results = []
    print("  " + "mode".ljust(10) + "recall(严格)".ljust(14) + "recall(宽)".ljust(13)
          + "precision".ljust(12) + "F1".ljust(9) + "全命中".ljust(9) + "p50")
    for mode in modes:
        r = eval_mode(mode, cases, args.top_k)
        results.append(r)
        print(f"  {mode.ljust(10)}{r['recall']:<14.3f}{r['recall_loose']:<13.3f}"
              f"{r['precision']:<12.3f}{r['f1']:<9.3f}{r['full_hit_rate']:<9.3f}"
              f"{r['p50_latency_ms']:.0f}ms")
    print("\n  严格口径 = 只有「小节标题就在讲这个配置项」的 chunk 才算命中；"
          "宽口径 = 提到即算。\n  宽口径在本语料上会虚高（见 README 指标口径说明），"
          "下方分析一律以严格口径为准。")

    print("\n分 hop 召回率（hop=2 的提升说明多跳关系确实在起作用）：")
    hops = sorted({h for r in results for h in r["recall_by_hop"]})
    header = "  " + "mode".ljust(10) + "".join(f"hop={h}".ljust(10) for h in hops)
    print(header)
    for r in results:
        row = "  " + r["mode"].ljust(10)
        row += "".join(f"{r['recall_by_hop'].get(h, float('nan')):.3f}".ljust(10) for h in hops)
        print(row)

    base = next((r for r in results if r["mode"] == "vector"), None)
    best = next((r for r in results if r["mode"] == "graphrag"), None)
    if base and best:
        print(f"\n  GraphRAG vs 朴素向量 RAG：召回 {base['recall']:.3f} → {best['recall']:.3f} "
              f"（{(best['recall'] - base['recall']) / max(base['recall'], 1e-9):+.1%}），"
              f"延迟 {base['p50_latency_ms']:.0f}ms → {best['p50_latency_ms']:.0f}ms")

    if args.verbose:
        print("\n漏召回明细（graphrag 模式）：")
        for c in (best or results[-1])["per_case"]:
            if c.get("missed"):
                print(f"  [{c['id']}] {c.get('query', '')}\n      漏：{c['missed']}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"top_k": args.top_k, "n_cases": len(cases), "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {OUT_PATH.relative_to(settings.root_dir)}")


if __name__ == "__main__":
    main()
