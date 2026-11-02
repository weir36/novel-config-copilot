"""追问识别评测。

每组是一段多轮对话，第 2 轮起标注了这一句是不是接着上文问（followup）。
评测时按顺序把每一轮真的走一遍在线链路（关掉 Agent，不需要 API key），
看 handle_chat 返回的 followup 判断对不对。

为什么要真的走链路而不是直接调 detect_followup：
追问识别依赖上一轮存进会话里的"话题焦点"，以及图上的邻居关系，
只有完整跑一遍，才能测到"上一轮存错了焦点、这一轮跟着判错"这类问题。

两份题库：tune 是设计判断规则时对照的，test 是规则定稿后才写的，只测不改。

用法：
    python -m eval.run_followup_eval              # 测试集
    python -m eval.run_followup_eval --split tune --verbose
"""
from __future__ import annotations

import argparse
import json
import uuid

from app.chat.pipeline import handle_chat
from app.foundation.models import ChatRequest
from app.foundation.settings import settings
from app.session.session import get_session_manager

SPLITS = {
    "tune": settings.gold_dir / "followup_tune.jsonl",
    "test": settings.gold_dir / "followup_eval.jsonl",
}
RESULTS_DIR = settings.root_dir / "eval" / "results"


def load_dialogs(path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if "_comment" not in obj:
                out.append(obj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dialogs = load_dialogs(SPLITS[args.split])
    sm = get_session_manager()
    tp = fp = fn = tn = 0
    rows: list[dict] = []

    for d in dialogs:
        sid = f"fu-{uuid.uuid4().hex[:8]}"
        sm.reset(sid)
        for i, turn in enumerate(d["turns"]):
            resp = handle_chat(ChatRequest(session_id=sid, message=turn["q"], use_agent=False))
            if i == 0:
                continue
            gold = bool(turn["followup"])
            pred = bool(resp.followup and resp.followup.is_followup)
            tp += gold and pred
            fp += (not gold) and pred
            fn += gold and not pred
            tn += (not gold) and not pred
            rows.append({
                "dialog": d["id"], "turn": i + 1, "query": turn["q"],
                "gold": gold, "pred": pred, "ok": gold == pred,
                "reasons": resp.followup.reasons if resp.followup else [],
            })

    n = len(rows)
    acc = (tp + tn) / n if n else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    print(f"追问识别评测（{args.split}）：{len(dialogs)} 组对话，{n} 个待判断的轮次\n")
    print(f"  准确率   {acc:.4f}  ({tp + tn}/{n})")
    print(f"  追问类   P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"  误判为追问 {fp} 个，漏判追问 {fn} 个")

    if args.verbose:
        wrong = [x for x in rows if not x["ok"]]
        if wrong:
            print(f"\n  错误样本（{len(wrong)} 个）：")
            for x in wrong:
                print(f"    [{x['dialog']} 第{x['turn']}轮]「{x['query']}」 gold={x['gold']} "
                      f"pred={x['pred']} 信号={x['reasons']}")

    out_path = RESULTS_DIR / ("followup_eval.json" if args.split == "test" else "followup_tune.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "n_dialogs": len(dialogs), "n_turns": n,
        "accuracy": round(acc, 4), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "false_positive": fp, "false_negative": fn, "details": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path.relative_to(settings.root_dir)}")


if __name__ == "__main__":
    main()
