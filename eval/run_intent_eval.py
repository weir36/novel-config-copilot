"""意图识别评测。

除了总体准确率，重点看两个东西：
  1. **混淆矩阵** —— 哪两类在互相错。explain/optimize 之间的混淆是这个任务的主要难点，
     因为"B004 的完本奖为什么开不了"（解释）和"B004 的配置有问题帮我改"（优化）都带着具体作品。
  2. **out_of_scope 的召回** —— 这一类漏判的代价最高：漏一个就多跑一整条
     RAG + Agent 链路。所以它的 recall 比整体 accuracy 更值得盯。
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from app.foundation.settings import settings
from app.intent.classifier import get_classifier
from app.foundation.models import Intent

# 两份题库：tune 是调参集（调种子样本和关键词时看它），test 是测试集（只在最后测一次）。
# 混着用的话，测出来的数字就是"对着考题改答案"的结果，没有参考价值。
SPLITS = {
    "tune": settings.gold_dir / "intent_tune.jsonl",
    "test": settings.gold_dir / "intent_eval.jsonl",
}
EVAL_PATH = SPLITS["test"]
RESULTS_DIR = settings.root_dir / "eval" / "results"

LABELS = [i.value for i in Intent]


def load_cases(path=EVAL_PATH) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_comment" not in obj:
            out.append(obj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--split", choices=list(SPLITS), default="test",
                    help="tune=调参集，test=测试集（默认）")
    args = ap.parse_args()

    cases = load_cases(SPLITS[args.split])
    out_path = RESULTS_DIR / ("intent_eval.json" if args.split == "test" else "intent_tune.json")
    clf = get_classifier()

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct = 0
    latencies: list[float] = []
    errors: list[dict] = []
    fallbacks = 0

    for case in cases:
        res = clf.classify(case["query"])
        pred, gold = res.intent.value, case["label"]
        confusion[gold][pred] += 1
        latencies.append(res.latency_ms)
        fallbacks += int(res.fallback_used)
        if pred == gold:
            correct += 1
        else:
            errors.append({
                "query": case["query"], "gold": gold, "pred": pred,
                "confidence": res.confidence, "scores": res.scores,
            })

    acc = correct / len(cases)
    print(f"意图识别评测（{args.split}）：{len(cases)} 条用例，分类器={clf.name}\n")
    print(f"  准确率        {acc:.4f}  ({correct}/{len(cases)})")
    print(f"  平均延迟      {statistics.mean(latencies):.1f} ms")
    print(f"  低置信兜底率  {fallbacks / len(cases):.2%}（兜底到 explain）")

    # 逐类 P/R/F1
    print("\n  逐类指标：")
    print("  " + "label".ljust(15) + "P".ljust(9) + "R".ljust(9) + "F1".ljust(9) + "support")
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        pred_total = sum(confusion[g][label] for g in LABELS)
        p = tp / pred_total if pred_total else 0.0
        r = tp / support if support else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f1, 4), "support": support}
        if support:
            print(f"  {label.ljust(15)}{p:<9.3f}{r:<9.3f}{f1:<9.3f}{support}")

    print("\n  混淆矩阵（行=真实，列=预测）：")
    print("  " + "".ljust(15) + "".join(l[:6].ljust(9) for l in LABELS))
    for gold in LABELS:
        if not sum(confusion[gold].values()):
            continue
        row = "  " + gold.ljust(15)
        row += "".join(str(confusion[gold][p]).ljust(9) for p in LABELS)
        print(row)

    oos = per_label.get("out_of_scope", {})
    if oos.get("support"):
        print(f"\n  out_of_scope 召回率 {oos['recall']:.3f} —— "
              f"每漏判一条就多跑一次完整 RAG+Agent 链路，这个指标比总体准确率更关键")

    if args.verbose and errors:
        print(f"\n  错误样本（{len(errors)} 条）：")
        for e in errors:
            print(f"    「{e['query']}」  gold={e['gold']}  pred={e['pred']}  "
                  f"conf={e['confidence']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "split": args.split,
            "classifier": clf.name, "n_cases": len(cases), "accuracy": round(acc, 4),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "fallback_rate": round(fallbacks / len(cases), 4),
            "per_label": per_label,
            "confusion": {g: dict(confusion[g]) for g in LABELS},
            "errors": errors,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path.relative_to(settings.root_dir)}")


if __name__ == "__main__":
    main()
