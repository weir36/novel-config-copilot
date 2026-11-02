"""端到端评测。

断言全部**程序可判定**，不用 LLM 裁判做主指标。原因是 LLM 裁判本身带 5~10 个点的
方差，用它当主指标会导致"改了 prompt 指标涨了 3 个点"这种结论根本不可信 ——
分不清是真的变好了还是裁判抖了。

四个维度：
  1. 意图正确率        —— 端到端链路里的意图判定是否正确
  2. 关键信息覆盖率     —— 回答里是否提到了必须提到的配置项 / 规则编号 / 关键词
  3. **方案合法率**     —— 推荐/优化类回答抽出的配置方案，能否通过规则引擎校验。
                          这是这个项目最硬的一个指标：LLM 可以说得天花乱坠，
                          但方案违反 R001 就是违反，没有解释余地。
  4. 短路正确率        —— out_of_scope 是否真的没进检索和 Agent

LLM 裁判作为可选的补充维度（--judge），只评可读性这类主观项，不进主指标。

**对照实验（--baseline）**：同一组用例，让「纯大模型」直接回答 —— 不检索、不走 Agent、
不过规则引擎，只在提示词里给它配置项清单。出来的方案照样交给规则引擎判对错。
完整系统和纯大模型的方案合法率之差，就是「检索 + 规则兜底」带来的配错减少幅度，
这是个人项目在没有真实用户的情况下，最接近"线上配错减少了多少"的可复现指标。

用法：
    python -m eval.run_e2e_eval                    # 需要 LLM_API_KEY
    python -m eval.run_e2e_eval --no-agent         # 关掉 Agent 做对照，看 Agent 的增量
    python -m eval.run_e2e_eval --baseline         # 纯大模型对照组
    python -m eval.run_e2e_eval --judge            # 额外跑 LLM 裁判
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

from app.agent.tools import detect_scope, validate_with_base
from app.validation.plan_extract import extract_plan
from app.chat.pipeline import handle_chat
from app.foundation.registry import get_registry
from app.foundation.settings import settings
from app.generation.llm_client import STATS, chat_json, get_llm, llm_meta
from app.generation.prompts import BASELINE_SYSTEM
from app.foundation.models import ChatRequest, Intent

EVAL_PATH = settings.gold_dir / "e2e_eval.jsonl"
RESULTS_DIR = settings.root_dir / "eval" / "results"

JUDGE_PROMPT = """你是一个严格的评审。下面是编辑的提问和助手的回答。
请从三个维度打分（1~5 分，只输出 JSON）：

- structure：结构是否清晰、是否按要求的格式组织
- actionable：编辑能否照着直接操作，有没有停留在空泛建议
- honesty：是否存在编造的取值/规则，或在资料不足时硬答

输出：{{"structure": n, "actionable": n, "honesty": n, "comment": "一句话理由"}}

【问题】{question}

【回答】{answer}"""


def load_cases() -> list[dict]:
    out = []
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_comment" not in obj:
            out.append(obj)
    return out


def check_mentions(answer: str, case: dict) -> tuple[int, int, list[str]]:
    """关键信息覆盖：配置项 key（key 或中文名都算）、规则编号、指定关键词。"""
    reg = get_registry()
    low = answer.lower()
    need, hit, missed = 0, 0, []

    for key in case.get("must_mention_keys", []):
        need += 1
        item = reg.items.get(key)
        surfaces = [key.lower()] + ([item.name.lower()] if item else [])
        if any(s in low for s in surfaces):
            hit += 1
        else:
            missed.append(f"key:{key}")

    for rule_id in case.get("must_mention_rules", []):
        need += 1
        if rule_id.lower() in low:
            hit += 1
        else:
            missed.append(f"rule:{rule_id}")

    for text in case.get("must_mention_text", []):
        need += 1
        if text.lower() in low:
            hit += 1
        else:
            missed.append(f"text:{text}")

    return hit, need, missed


def check_plan(answer: str, case: dict) -> dict:
    """方案合法性检查。"""
    out = {"applicable": False, "passed": None, "violations": [], "forbidden_hit": [],
           "expected_missing": []}
    if not (case.get("plan_must_pass") or case.get("forbidden_values")
            or case.get("expect_values")):
        return out
    out["applicable"] = True

    plan = extract_plan(answer, use_llm_fallback=False)
    if not plan.values:
        out["passed"] = False
        out["violations"] = ["未能从回答中抽取出任何配置方案"]
        return out

    # 和线上同一套口径：合并问题里那本书 / 作者的当前配置再校验
    report, _ = validate_with_base(plan, detect_scope(case["query"]))
    out["passed"] = report.passed
    out["violations"] = [f"[{v.rule_id}] {v.message}" for v in report.errors]

    for key, bad in (case.get("forbidden_values") or {}).items():
        if plan.values.get(key) == bad:
            out["forbidden_hit"].append(f"{key}={bad}")
    for key, want in (case.get("expect_values") or {}).items():
        if plan.values.get(key) != want:
            out["expected_missing"].append(f"{key} 期望 {want}，实际 {plan.values.get(key)}")

    return out


def baseline_answer(question: str) -> str:
    """纯大模型对照组：不检索、不走 Agent、不校验，只给配置项清单。"""
    reg = get_registry()
    item_keys = "\n".join(
        f"- {k}（{i.name}）类型={i.type}" + (f" 可选={i.options}" if i.options else "")
        for k, i in reg.items.items()
    )
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM.format(item_keys=item_keys)},
        {"role": "user", "content": question},
    ]
    return get_llm().chat(messages, purpose="baseline", max_tokens=2000)


def judge(question: str, answer: str) -> dict:
    try:
        data = chat_json(
            [{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer[:4000])}],
            purpose="judge", temperature=0.0, max_tokens=400,
        )
        return {k: data.get(k) for k in ("structure", "actionable", "honesty", "comment")}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-agent", action="store_true", help="关闭 Agent 做对照组")
    ap.add_argument("--judge", action="store_true", help="额外跑 LLM 裁判（主观维度）")
    ap.add_argument("--baseline", action="store_true",
                    help="纯大模型对照组：不检索、不走 Agent、不过规则引擎")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    meta = llm_meta()
    if meta["provider"] == "mock":
        print("⚠️  当前为 MockLLM（未配置 LLM_API_KEY），端到端指标无意义。\n"
              "    仅用于验证链路可跑通。配置 .env 后重跑可得真实指标。\n")

    use_agent = not args.no_agent and not args.baseline
    mode = "baseline" if args.baseline else ("agent" if use_agent else "no_agent")
    print(f"端到端评测：{len(cases)} 条用例  |  LLM={meta['model']}  |  模式={mode}\n")

    intent_ok = 0
    mention_hits, mention_needs = 0, 0
    plan_cases, plan_passed = 0, 0
    short_circuit_ok, short_circuit_cases = 0, 0
    latencies: list[float] = []
    agent_steps: list[int] = []
    details: list[dict] = []

    for case in cases:
        t0 = time.perf_counter()
        gold_intent = case["intent"]
        if args.baseline:
            # 对照组没有意图识别和检索，越界题也照答，所以意图和短路两项不统计
            answer, pred_intent, retrieval, agent_trace = (
                baseline_answer(case["query"]), gold_intent, None, None)
        else:
            resp = handle_chat(ChatRequest(
                session_id=f"eval-{uuid.uuid4().hex[:8]}",
                message=case["query"],
                use_agent=use_agent,
            ))
            answer, pred_intent = resp.answer, resp.intent.intent.value
            retrieval, agent_trace = resp.retrieval, resp.agent_trace
        latencies.append((time.perf_counter() - t0) * 1000)

        intent_correct = pred_intent == gold_intent
        intent_ok += int(intent_correct)

        hit, need, missed = check_mentions(answer, case)
        mention_hits += hit
        mention_needs += need

        plan_res = check_plan(answer, case)
        if plan_res["applicable"]:
            plan_cases += 1
            ok = (
                plan_res["passed"]
                and not plan_res["forbidden_hit"]
                and not plan_res["expected_missing"]
            )
            plan_passed += int(bool(ok))

        if case.get("expect_short_circuit") and not args.baseline:
            short_circuit_cases += 1
            # 短路的判定：没有检索结果、没有 Agent 轨迹
            ok = retrieval is None and agent_trace is None
            short_circuit_ok += int(ok)

        if agent_trace:
            agent_steps.append(len(agent_trace.steps))

        row = {
            "id": case["id"], "query": case["query"],
            "intent_gold": gold_intent, "intent_pred": pred_intent,
            "intent_ok": intent_correct,
            "mention": f"{hit}/{need}", "missed": missed,
            "plan": plan_res,
            "agent_steps": len(agent_trace.steps) if agent_trace else 0,
            "latency_ms": round(latencies[-1], 1),
            "note": case.get("note", ""),
        }
        if args.judge:
            row["judge"] = judge(case["query"], answer)
        details.append(row)

        flag = "✓" if intent_correct and not missed else "✗"
        print(f"  {flag} [{case['id']}] {case['query'][:28]:30s} "
              f"意图 {pred_intent:12s} 覆盖 {hit}/{need}"
              + (f"  方案{'通过' if plan_res['passed'] else '未通过'}"
                 if plan_res["applicable"] else ""))

    n = len(cases)
    summary = {
        "n_cases": n,
        "mode": mode,
        "agent_enabled": use_agent,
        "llm": meta,
        "intent_accuracy": None if args.baseline else round(intent_ok / n, 4),
        "mention_coverage": round(mention_hits / mention_needs, 4) if mention_needs else 0.0,
        "plan_valid_rate": round(plan_passed / plan_cases, 4) if plan_cases else None,
        "plan_cases": plan_cases,
        "short_circuit_rate": (
            round(short_circuit_ok / short_circuit_cases, 4) if short_circuit_cases else None
        ),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(n * 0.95) - 1], 1),
        "avg_agent_steps": round(statistics.mean(agent_steps), 2) if agent_steps else 0,
        "llm_stats": STATS.snapshot(),
    }

    if summary["intent_accuracy"] is not None:
        print(f"\n  意图正确率      {summary['intent_accuracy']:.4f}")
    print(f"  关键信息覆盖率   {summary['mention_coverage']:.4f}  "
          f"({mention_hits}/{mention_needs})")
    if plan_cases:
        print(f"  方案合法率      {summary['plan_valid_rate']:.4f}  "
              f"({plan_passed}/{plan_cases})　← 规则引擎判定，无解释余地")
    if short_circuit_cases:
        print(f"  越界短路正确率   {summary['short_circuit_rate']:.4f}")
    print(f"  延迟 p50/p95    {summary['p50_latency_ms']:.0f} / "
          f"{summary['p95_latency_ms']:.0f} ms")
    if agent_steps:
        print(f"  平均 Agent 步数 {summary['avg_agent_steps']:.2f}")
    print(f"  LLM 调用        {STATS.calls} 次，"
          f"prompt {STATS.prompt_tokens} tok / completion {STATS.completion_tokens} tok")

    if args.judge:
        for dim in ("structure", "actionable", "honesty"):
            vals = [d["judge"].get(dim) for d in details
                    if isinstance(d.get("judge"), dict) and isinstance(d["judge"].get(dim), (int, float))]
            if vals:
                print(f"  裁判-{dim:11s} {statistics.mean(vals):.2f} / 5")

    out_path = RESULTS_DIR / {"agent": "e2e_eval.json", "no_agent": "e2e_eval_no_agent.json",
                              "baseline": "e2e_eval_baseline.json"}[mode]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path.relative_to(settings.root_dir)}")


if __name__ == "__main__":
    main()
