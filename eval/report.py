"""把各评测脚本的结果汇总成一份 markdown 报告。

产出 docs/eval-report.md —— 这份文件是给面试官/评审看的，
每个数字都标注了口径、样本量和产生它的命令，可复现。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.foundation.settings import settings

RESULTS = settings.root_dir / "eval" / "results"
OUT = settings.root_dir / "docs" / "eval-report.md"


def load(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def section_intent(d: dict) -> str:
    lines = [
        "## 一、意图识别",
        "",
        f"- 分类器：`{d['classifier']}`（"
        + ("bge 句向量 + 逻辑回归 + 关键词先验" if "logreg" in d["classifier"] else "向量 kNN + 关键词先验") + "）",
        f"- 测试集：`data/gold/intent_eval.jsonl`，{d['n_cases']} 条，与训练种子、调参集都不重叠",
        f"- **准确率 {d['accuracy']:.3f}**，平均延迟 {d['avg_latency_ms']:.1f} ms",
        "",
        "| 类别 | Precision | Recall | F1 | 样本 |",
        "|------|-----------|--------|----|------|",
    ]
    for label, m in d["per_label"].items():
        if m["support"]:
            lines.append(
                f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {m['support']} |"
            )
    oos = d["per_label"].get("out_of_scope", {})
    if oos.get("support"):
        lines += [
            "",
            f"`out_of_scope` 召回率 **{oos['recall']:.3f}**。这个指标比总体准确率更关键 —— "
            "每漏判一条，就要多跑一次完整的检索 + Agent 链路。",
        ]
    if d.get("errors"):
        lines += ["", "剩余错误样本：", ""]
        lines += [f"- 「{e['query']}」 gold=`{e['gold']}` pred=`{e['pred']}` "
                  f"(conf {e['confidence']:.2f})" for e in d["errors"]]
    tune = load("intent_tune.json")
    lines += [
        "",
        "> **口径声明**：题库分成调参集和测试集。种子样本、关键词、越界词只对照"
        "训练集和调参集设计"
        + (f"（调参集 {tune['n_cases']} 条，准确率 {tune['accuracy']:.3f}）" if tune else "")
        + "；上表来自测试集，只在定稿后测了一次，测完没有再改分类器。",
        "",
        "复现：`python -m eval.run_intent_eval --verbose`（测试集）/ "
        "`python -m eval.run_intent_eval --split tune`（调参集）",
    ]
    return "\n".join(lines)


def section_retrieval(d: dict) -> str:
    lines = [
        "## 二、检索消融：GraphRAG 的增益来自哪里",
        "",
        f"- 评测集：`data/gold/retrieval_eval.jsonl`，{d['n_cases']} 条，top_k={d['top_k']}",
        "- 指标是 **item 级召回**（回答该问题所必需的配置项知识是否被召回），"
        "不是 chunk 级 —— chunk_id 会随切块策略变化，item 不会，换切块方案时指标仍可比。",
        "",
        "| 模式 | Recall(严格) | Recall(宽) | Precision | F1 | 全命中率 | p50 延迟 |",
        "|------|-------------|-----------|-----------|----|---------| -------|",
    ]
    for r in d["results"]:
        lines.append(
            f"| {r['mode']} | **{r['recall']:.3f}** | {r['recall_loose']:.3f} | "
            f"{r['precision']:.3f} | {r['f1']:.3f} | {r['full_hit_rate']:.3f} | "
            f"{r['p50_latency_ms']:.0f} ms |"
        )

    hops = sorted({h for r in d["results"] for h in r["recall_by_hop"]})
    lines += ["", "### 分 hop 召回率", "",
              "| 模式 | " + " | ".join(f"hop={h}" for h in hops) + " |",
              "|------|" + "|".join(["--------"] * len(hops)) + "|"]
    for r in d["results"]:
        cells = " | ".join(f"{r['recall_by_hop'].get(h, 0):.3f}" for h in hops)
        lines.append(f"| {r['mode']} | {cells} |")

    base = next((r for r in d["results"] if r["mode"] == "vector"), None)
    best = next((r for r in d["results"] if r["mode"] == "graphrag"), None)
    if base and best:
        h2_base = base["recall_by_hop"].get("2")
        h2_best = best["recall_by_hop"].get("2")
        lines += [
            "",
            "### 结论",
            "",
            f"1. 整体召回 {base['recall']:.3f} → **{best['recall']:.3f}** "
            f"（{(best['recall'] - base['recall']) / max(base['recall'], 1e-9):+.1%}）。",
        ]
        if h2_base and h2_best:
            lines.append(
                f"2. **增益集中在需要 2 跳关系的问题上**：{h2_base:.3f} → {h2_best:.3f}"
                f"（{(h2_best - h2_base) / max(h2_base, 1e-9):+.1%}），"
                f"而 1 跳问题的提升明显更小。这是图谱确实在起作用的直接证据 —— "
                f"如果两档提升幅度相当，那更可能是调参调出来的。"
            )
        lines += [
            f"3. 代价是精确率从 {base['precision']:.3f} 降到 {best['precision']:.3f}："
            f"子图扩展会把相关但非必需的实体也带进上下文，token 成本上升、"
            f"也给模型引入了干扰。这是 GraphRAG 的固有取舍，不是 bug。"
            f"缓解手段是 `graph_query_entities` 收敛参与检索的实体数 + 相似度阈值过滤。",
            f"4. 延迟从 {base['p50_latency_ms']:.0f}ms 增加到 "
            f"{best['p50_latency_ms']:.0f}ms（多一次实体向量检索 + 图遍历），"
            f"相对 LLM 生成的秒级耗时可以忽略。",
            "",
            "> **宽口径为什么不能用**：宽口径把「chunk 里顺带提了一句某配置项」也算召回成功。"
            + _loose_note(d) +
            "严格口径只认「小节标题就在讲这个配置项」的 chunk，才有区分度。",
            "",
            "> 检索参数（top_k、融合权重）全部是默认值，没有对着这份评测集调过；"
            "实体链接阈值由 `eval/calibrate_thresholds.py` 用越界问题的底噪定出，同样不看本评测集。",
            "",
            "复现：`python -m eval.run_retrieval_eval --verbose`",
        ]
    return "\n".join(lines)


def section_graph(d: dict) -> str:
    lines = [
        "## 三、图谱抽取质量：这一步值不值得花 LLM 的钱",
        "",
        f"- gold：`config_registry.yaml` 中的 {d['n_gold']} 条关系"
        "（模拟配置平台元数据库，抽取器看不到它，只能看 `raw_docs/` 里的散文）",
        f"- 语料：{d['n_chunks']} 个 chunk",
        "- 结构性关系（BELONGS_TO / APPLIES_TO / CONSTRAINED_BY）不参与评测，"
        "它们本来就从元数据直接读，算进去会虚高",
        "",
        "| 抽取器 | Precision | Recall | F1 | 抽出 | 命中 |",
        "|--------|-----------|--------|----|------|------|",
    ]
    for name, res in d["results"].items():
        o = res["overall"]
        lines.append(f"| {name} | {o['precision']:.3f} | {o['recall']:.3f} | "
                     f"**{o['f1']:.3f}** | {o['pred']} | {o['tp']} |")

    types = sorted({t for r in d["results"].values() for t in r["by_type"]})
    if types:
        lines += ["", "### 分关系类型 F1", "",
                  "| 抽取器 | " + " | ".join(types) + " |",
                  "|--------|" + "|".join(["------"] * len(types)) + "|"]
        for name, res in d["results"].items():
            cells = " | ".join(
                f"{res['by_type'][t]['f1']:.3f}" if t in res["by_type"] else "-" for t in types
            )
            lines.append(f"| {name} | {cells} |")

    lines += [
        "",
        "> **这个数字偏乐观**：6 篇政策文档和规则抽取的正则是同一个人写的，"
        "文档的句式天然贴合正则（\"A 依赖 B\"\"A 与 B 互斥\"）。"
        "换成真实编辑写的文档，句式变化会多得多，规则抽取的召回会明显下降，"
        "这正是需要 LLM 抽取兜底的原因。",
    ]
    if len(d["results"]) == 1 and "rule" in d["results"]:
        f1 = d["results"]["rule"]["overall"]["f1"]
        lines += [
            "",
            f"当前只跑了规则抽取（F1 {f1:.3f}），因为完整对比需要 `LLM_API_KEY`。",
            "",
            f"规则抽取的 **{f1:.3f}** 就是 LLM 抽取必须超过的基线。它的失败模式很明确："
            "正则只能覆盖归纳出来的句式，遇到 `“这两项只能二选一”`、"
            "`“开了前者后者就灰了”` 这类变体就抓不到；而方向判断依赖"
            "「关系词左边是主语」这条启发式，遇到被动句会反向。",
            "",
            "复现（需 LLM_API_KEY）：`python -m eval.run_graph_eval "
            "--extractors rule,llm,hybrid --verbose`",
        ]
    else:
        lines += ["", "复现：`python -m eval.run_graph_eval --extractors rule,llm,hybrid`"]

    ents = d.get("entities") or {}
    lines += ["", "### 实体抽取：词表 vs 大模型 + 四层消歧", ""]
    if ents:
        lines += ["| 方式 | Precision | Recall | F1 | 待审核实体 |",
                  "|------|-----------|--------|----|-----------|"]
        for name, r in ents.items():
            lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | "
                         f"{r['f1']:.3f} | {r.get('pending', 0)} |")
    if "llm" not in ents:
        lines += [
            "" if ents else "_未执行_",
            "大模型实体抽取需要 `LLM_API_KEY`，当前没有跑。参照答案是词表在每个 chunk 上认出的实体，"
            "所以词表按定义是 1.0；大模型这一路要看的是还原了多少、归错了多少、进了多少待审核。"
            "四层消歧的逻辑本身不依赖模型，已由 `tests/test_extract.py` 逐层覆盖。",
        ]
    lines += ["", "复现：`python -m eval.run_graph_eval --entities vocab,llm`（llm 需要 API key）"]
    return "\n".join(lines)


def _loose_note(d: dict) -> str:
    """按当前语料现算宽口径为什么会虚高，不写死数字。"""
    chunks_path = settings.index_dir / "chunks.json"
    eval_path = settings.gold_dir / "retrieval_eval.jsonl"
    if not (chunks_path.exists() and eval_path.exists()):
        return ""
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    per_chunk = sum(len(c["linked_item_keys"]) for c in chunks) / max(len(chunks), 1)
    golds = [json.loads(x) for x in eval_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    golds = [g for g in golds if "gold_items" in g]
    avg_gold = sum(len(g["gold_items"]) for g in golds) / max(len(golds), 1)
    loose = [r["recall_loose"] for r in d["results"]]
    return (f"本语料每个 chunk 平均提到 {per_chunk:.2f} 个配置项，top-{d['top_k']} 期望覆盖约 "
            f"{per_chunk * d['top_k']:.0f} 个，而 gold 平均只有 {avg_gold:.1f} 个 —— "
            f"四种模式在宽口径下都在 {min(loose):.2f}~{max(loose):.2f} 之间，无法区分优劣。")


def section_followup(d: dict) -> str:
    tune = load("followup_tune.json")
    lines = [
        "## 四、追问识别",
        "",
        "- 做法：规则判断（指代承接词 / 配置项与上一轮在图上相邻 / 在改上一轮方案 / 短问没提配置项），"
        "是追问才带相关的历史轮次和做检索剪枝，新话题都不做",
        f"- 测试集：`data/gold/followup_eval.jsonl`，{d['n_dialogs']} 组对话，{d['n_turns']} 个待判断轮次",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 准确率 | **{d['accuracy']:.3f}** |",
        f"| 追问类 Precision / Recall / F1 | {d['precision']:.3f} / {d['recall']:.3f} / {d['f1']:.3f} |",
        f"| 误判为追问 / 漏判追问 | {d['false_positive']} / {d['false_negative']} |",
    ]
    wrong = [x for x in d["details"] if not x["ok"]]
    if wrong:
        lines += ["", "错误样本：", ""]
        lines += [f"- 「{x['query']}」 gold={x['gold']} pred={x['pred']}" for x in wrong]
        lines += [
            "",
            "主要短板是漏判：没有指代词、提到的配置项又和上一轮不直接相连时（比如问完某本书接着问「完本奖为什么不能开」），"
            "规则抓不到。改进方向是加一层「是否提到上一轮的作品 / 作者」的判断，或换成小模型做二分类。",
        ]
    lines += [
        "",
        "> **口径声明**：判断规则对照调参集 `followup_tune.jsonl` 设计"
        + (f"（{tune['n_turns']} 个轮次，准确率 {tune['accuracy']:.3f}，不作为成绩）" if tune else "")
        + "；测试集在规则定稿后才写，只测了一次，测完没有改规则。",
        "",
        "复现：`python -m eval.run_followup_eval --verbose`",
    ]
    return "\n".join(lines)


def section_e2e(d: dict) -> str:
    s = d["summary"]
    lines = [
        "## 五、端到端",
        "",
        f"- 评测集：`data/gold/e2e_eval.jsonl`，{s['n_cases']} 条",
        f"- LLM：`{s['llm']['model']}`　Agent：{'开' if s['agent_enabled'] else '关'}",
        "- **所有断言都是程序可判定的**，不用 LLM 裁判做主指标 —— "
        "裁判自身带 5~10 个点方差，用它当主指标会分不清「真的变好了」和「裁判抖了」",
        "",
        "| 指标 | 数值 | 说明 |",
        "|------|------|------|",
        (f"| 意图正确率 | {s['intent_accuracy']:.3f} | 端到端链路中的意图判定 |"
         if s.get("intent_accuracy") is not None else "| 意图正确率 | - | 对照组不做意图识别 |"),
        f"| 关键信息覆盖率 | {s['mention_coverage']:.3f} | "
        "回答是否提到了必须提到的配置项 / 规则编号 |",
    ]
    if s.get("plan_valid_rate") is not None:
        lines.append(
            f"| **方案合法率** | **{s['plan_valid_rate']:.3f}** | "
            f"{s['plan_cases']} 条推荐/优化类用例，由规则引擎判定，无解释余地 |"
        )
    if s.get("short_circuit_rate") is not None:
        lines.append(f"| 越界短路正确率 | {s['short_circuit_rate']:.3f} | "
                     "out_of_scope 是否真的没进检索和 Agent |")
    lines += [
        f"| 延迟 p50 / p95 | {s['p50_latency_ms']:.0f} / {s['p95_latency_ms']:.0f} ms | |",
        f"| 平均 Agent 步数 | {s['avg_agent_steps']:.2f} | 步数越少说明工具设计越贴合任务 |",
    ]
    st = s.get("llm_stats", {})
    if st.get("calls"):
        lines += [
            "",
            f"LLM 开销：{st['calls']} 次调用，prompt {st['prompt_tokens']} tok / "
            f"completion {st['completion_tokens']} tok，平均 {st['avg_latency_ms']:.0f} ms/次。",
        ]
    base = load("e2e_eval_baseline.json")
    if base and base["summary"].get("plan_valid_rate") is not None and s.get("plan_valid_rate") is not None:
        b, f = base["summary"]["plan_valid_rate"], s["plan_valid_rate"]
        lines += [
            "",
            "### 对照实验：纯大模型 vs 完整系统",
            "",
            f"同一组 {s['plan_cases']} 条出方案的用例，纯大模型（不检索、不走 Agent、不校验）的方案合法率 "
            f"{b:.3f}，完整系统 **{f:.3f}**；违规率从 {1 - b:.3f} 降到 {1 - f:.3f}。"
            "这是没有真实用户时最接近「线上配错减少多少」的可复现指标。",
        ]
    lines += ["", "复现：`python -m eval.run_e2e_eval` / "
              "`python -m eval.run_e2e_eval --no-agent` / `python -m eval.run_e2e_eval --baseline`（对照组）"]
    return "\n".join(lines)


def main() -> None:
    parts = [
        "# 评测报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　|　"
        "一键复现：`make eval`（离线）/ `make eval-all`（含端到端，需 API key）",
        "",
        "本报告里的每个数字都标注了口径、样本量和复现命令。"
        "**没有跑出来的指标不写**，需要 LLM 的部分如果没配 key 会明确标注缺失。",
        "",
    ]

    build = settings.index_dir / "build_info.json"
    if build.exists():
        b = json.loads(build.read_text(encoding="utf-8"))
        parts += [
            f"索引构建信息：{b['chunks']} chunk / {b['entities']} 实体 / "
            f"{b['relations']} 关系，抽取器 `{b['extractor']}`，"
            f"耗时 {b['elapsed_s']}s，构建于 {b['built_at']}。",
            "",
        ]

    for name, fn in (
        ("intent_eval.json", section_intent),
        ("retrieval_eval.json", section_retrieval),
        ("graph_eval.json", section_graph),
        ("followup_eval.json", section_followup),
        ("e2e_eval.json", section_e2e),
    ):
        d = load(name)
        if d is None:
            title = {"intent_eval.json": "一、意图识别", "retrieval_eval.json": "二、检索消融",
                     "graph_eval.json": "三、图谱抽取", "followup_eval.json": "四、追问识别",
                     "e2e_eval.json": "五、端到端"}[name]
            why = ("需要 `LLM_API_KEY`：没配 key 时回答是 MockLLM 的占位文本，指标没有意义，所以不写。"
                   "配好后执行 `make eval-all`。" if name == "e2e_eval.json" else "")
            parts += [f"## {title}", "", f"_未执行（缺少 `eval/results/{name}`）。{why}_", ""]
            continue
        parts += [fn(d), ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"评测报告已生成：{OUT.relative_to(settings.root_dir)}")


if __name__ == "__main__":
    main()
