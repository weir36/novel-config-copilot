"""Agent 工具集。

工具划分的原则：**把"必须精确"的事从 LLM 手里拿走**。

配置项的取值范围、依赖链、当前配置、约束是否满足、影响多大 ——
这五件事全都有确定性答案，让 LLM 靠记忆回答就是在制造幻觉。
所以它们各自是一个工具，LLM 只负责决定"现在该查什么"和"查到之后怎么办"。

每个工具返回两份东西：content（给 LLM 读的自然语言）和 data（给程序用的结构化结果）。
分开的原因是 Agent 循环里 LLM 只看 content，但最终响应要把 data 带给前端做渲染。
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from app.foundation.registry import get_registry
from app.foundation.settings import settings
from app.retrieval.graphrag import get_retriever
from app.validation.rule_engine import get_rule_engine
from app.foundation.models import ConfigPlan, ToolResult

TOOLS: dict[str, dict] = {}


def tool(name: str, description: str, parameters: dict):
    def deco(fn: Callable[..., tuple[str, dict]]):
        TOOLS[name] = {"fn": fn, "description": description, "parameters": parameters}
        return fn

    return deco


def _load_store() -> dict:
    path = settings.config_store_dir / "current.json"
    return json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# 工具实现
# =============================================================================


@tool(
    "search_knowledge",
    "在业务知识库中检索配置项的说明、规则、典型方案。当你需要了解某个配置项的定义、"
    "约束关系或业务背景时使用。",
    {
        "query": {"type": "string", "description": "检索问题，用完整的自然语言句子，不要只给关键词"},
    },
)
def search_knowledge(query: str) -> tuple[str, dict]:
    result = get_retriever().retrieve(query, mode="graphrag")
    text = result.context_text() or "未检索到相关资料。"
    return text[:4000], {
        "chunk_ids": [c.chunk.chunk_id for c in result.chunks],
        "entity_ids": [e.entity.entity_id for e in result.subgraph_entities],
        "triple_count": len(result.triples),
    }


@tool(
    "get_config_detail",
    "查询一个或多个配置项的精确定义：类型、取值范围、默认值、支持维度、依赖链、互斥项、业务影响。"
    "任何时候需要给出具体取值，都必须先调这个工具确认，不要凭记忆。",
    {
        "keys": {"type": "array", "description": "配置项的 key 或中文名，可传多个"},
    },
)
def get_config_detail(keys: list[str] | str) -> tuple[str, dict]:
    reg = get_registry()
    if isinstance(keys, str):
        keys = [keys]
    lines: list[str] = []
    found: dict[str, dict] = {}
    missing: list[str] = []

    for raw in keys:
        item = reg.resolve(raw)
        if item is None:
            missing.append(raw)
            continue
        chain = reg.dependency_chain(item.key)
        parts = [item.describe()]
        if chain:
            names = " → ".join(reg.items[k].name for k in [*chain, item.key] if k in reg.items)
            parts.append(f"依赖链：{names}")
        if item.requires_value:
            parts.append("取值依赖：" + "；".join(
                f"{reg.items[k].name} 须为 {v}" for k, v in item.requires_value.items()
                if k in reg.items
            ))
        if item.mutex_with:
            parts.append("互斥于：" + "、".join(
                reg.items[k].name for k in item.mutex_with if k in reg.items
            ))
        if item.enhances:
            parts.append("增强：" + "、".join(reg.effects.get(e, e) for e in item.enhances))
        if item.weakens:
            parts.append("削弱：" + "、".join(reg.effects.get(e, e) for e in item.weakens))
        rules = [
            r["id"] for r in reg.composite_rules
            if item.key in json.dumps(r, ensure_ascii=False)
        ]
        if rules:
            parts.append("受复合规则约束：" + "、".join(rules))
        lines.append("\n".join(parts))
        found[item.key] = {
            "name": item.name, "type": item.type, "default": item.default,
            "options": item.options, "range": item.range, "dimensions": item.dimensions,
        }

    if missing:
        lines.append(f"未找到以下配置项（可能拼写有误或不存在）：{'、'.join(missing)}")
    return "\n\n---\n\n".join(lines) or "未查到任何配置项。", {"found": found, "missing": missing}


@tool(
    "get_current_config",
    "查询某个频道 / 作者 / 作品当前的配置值。做「在现有配置上调整」类任务时必须先调用，"
    "否则你不知道当前状态就没法说清改了什么。",
    {
        "scope": {"type": "string", "description": "channel / author / book"},
        "id": {"type": "string", "description": "如 male、W1001、B003"},
    },
)
def get_current_config(scope: str, id: str) -> tuple[str, dict]:  # noqa: A002
    store = _load_store()
    bucket = {"channel": "channels", "author": "authors", "book": "books"}.get(scope)
    if bucket is None:
        return f"未知的 scope：{scope}，应为 channel/author/book。", {}
    entry = store.get(bucket, {}).get(id)
    if entry is None:
        available = list(store.get(bucket, {}).keys())
        return f"未找到 {scope}={id}。当前可用：{'、'.join(available)}", {"available": available}

    reg = get_registry()
    # 把继承链也带上：作品的生效配置 = 频道 < 作者 < 作品
    merged: dict[str, Any] = {}
    origin: dict[str, str] = {}
    chain: list[tuple[str, str, str | None]] = []
    if scope == "book":
        chain = [("channel", "channels", entry.get("channel")),
                 ("author", "authors", entry.get("author"))]
    elif scope == "author":
        chain = [("channel", "channels", entry.get("channel"))]
    for level, bkt, key in chain:
        vals = store.get(bkt, {}).get(key, {}).get("values", {})
        for k, v in vals.items():
            merged[k], origin[k] = v, level
    for k, v in entry.get("values", {}).items():
        merged[k], origin[k] = v, scope

    lines = [f"{entry.get('name', id)}（{scope}={id}）当前生效配置："]
    for k, v in merged.items():
        item = reg.items.get(k)
        label = item.name if item else k
        lines.append(f"- {label}（{k}）= {v}　[生效维度：{origin[k]}]")
    lines.append("\n注意：生效优先级 频道 < 作者 < 作品，上表已按优先级合并。")
    return "\n".join(lines), {"values": merged, "origin": origin, "meta": {
        k: v for k, v in entry.items() if k != "values"
    }}


_SCOPE_RE = (("book", re.compile(r"\bB\d{3}\b")), ("author", re.compile(r"\bW\d{4}\b")))


def detect_scope(text: str) -> str | None:
    """从问题里认出它在说哪本书 / 哪个作者，返回 book:B003 这样的基准范围。书优先于作者。"""
    for scope, pattern in _SCOPE_RE:
        if m := pattern.search(text or ""):
            return f"{scope}:{m.group(0)}"
    return None


def validate_with_base(plan: ConfigPlan, base_scope: str | None):
    """当前配置打底、方案覆盖，再交给规则引擎。返回 (报告, 合并后的取值)。

    优化类回答只列「变更项」：给 B003 开加更活动时，它依赖的全勤奖 B003 早就开着，不会再写一遍。
    只校验变更项会把这种合法方案误判成缺依赖；反过来，给买断书开完本奖这类「新配项 × 老配项」
    的冲突，也只有合并当前配置才查得出来。"""
    merged = dict(plan.values)
    if base_scope and ":" in base_scope:
        scope, sid = base_scope.split(":", 1)
        _txt, data = get_current_config(scope, sid)
        if data.get("values"):
            merged = {**data["values"], **merged}
    report = get_rule_engine().validate(plan.model_copy(update={"values": merged}))
    return report, merged


@tool(
    "check_constraints",
    "对一套配置方案做依赖、互斥、维度合法性与组合规则(R001~R004)的完整校验。"
    "在你给出任何配置方案之前**必须**调用一次，校验不通过要先改方案再重新校验。",
    {
        "values": {"type": "object", "description": "配置项 key -> 取值 的字典"},
        "dimensions": {"type": "object", "description": "可选，配置项 key -> 维度 key"},
        "base_scope": {"type": "string", "description": "可选，基准范围如 book:B003，"
                                                        "会把当前配置合并进来一起校验"},
    },
)
def check_constraints(
    values: dict, dimensions: dict | None = None, base_scope: str | None = None
) -> tuple[str, dict]:
    report, merged = validate_with_base(ConfigPlan(values=dict(values or {}), dimensions=dimensions or {}), base_scope)
    base_info = f"（已合并 {base_scope} 的当前配置一起校验）" if merged != (values or {}) else ""
    header = f"校验结果{base_info}：{'通过' if report.passed else '未通过'}"
    return f"{header}\n{report.to_text()}", {
        "passed": report.passed,
        "errors": [v.model_dump() for v in report.errors],
        "warns": [v.model_dump() for v in report.warns],
        "merged_values": merged,
    }


@tool(
    "estimate_impact",
    "试算一套配置变更对各业务指标（作者收入、读者留存、新书曝光、平台收益、更新稳定性）的相对影响。"
    "给出配置方案时用它来量化代价，不要空口说「会提升曝光」。",
    {
        "values": {"type": "object", "description": "本次要设置的配置项 key -> 取值"},
        "base_scope": {"type": "string", "description": "可选，对比基准如 book:B003"},
    },
)
def estimate_impact(values: dict, base_scope: str | None = None) -> tuple[str, dict]:
    store = _load_store()
    coeff: dict[str, dict] = store.get("effect_baseline", {})
    reg = get_registry()

    base: dict[str, Any] = {}
    if base_scope and ":" in base_scope:
        scope, sid = base_scope.split(":", 1)
        _txt, data = get_current_config(scope, sid)
        base = data.get("values", {})

    impact: dict[str, float] = {}
    details: list[str] = []
    for key, new_val in (values or {}).items():
        c = coeff.get(key)
        item = reg.items.get(key)
        if c is None or item is None:
            continue
        old_val = base.get(key, item.default)
        delta = _delta_factor(item, old_val, new_val)
        if abs(delta) < 1e-6:
            continue
        for effect, unit in c.items():
            impact[effect] = impact.get(effect, 0.0) + unit * delta
        details.append(
            f"- {item.name}：{old_val} → {new_val}　"
            + "、".join(
                f"{reg.effects.get(e, e)} {unit * delta:+.1%}" for e, unit in c.items()
            )
        )

    if not impact:
        return "本次变更未命中任何可量化的影响系数，无法给出试算。", {"impact": {}}

    lines = ["【影响试算】（相对当前基线的估计值，用于判断量级，非精确预测）"]
    for effect, v in sorted(impact.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"- {reg.effects.get(effect, effect)}：{v:+.1%}")
    lines.append("\n【逐项拆解】")
    lines.extend(details)
    return "\n".join(lines), {"impact": impact}


def _delta_factor(item, old_val: Any, new_val: Any) -> float:
    """把不同类型的取值变化折算成一个无量纲的变化幅度。"""
    if item.type == "bool":
        return float(bool(new_val)) - float(bool(old_val))
    if item.type == "enum":
        opts = item.options or []
        if old_val not in opts or new_val not in opts or len(opts) < 2:
            return 0.0 if old_val == new_val else 1.0
        return (opts.index(new_val) - opts.index(old_val)) / (len(opts) - 1)
    try:
        o, n = float(old_val or 0), float(new_val or 0)
    except (TypeError, ValueError):
        return 0.0
    if item.range and item.range[1] != item.range[0]:
        return (n - o) / (item.range[1] - item.range[0]) * 10  # 放大到与枚举可比的量级
    return n - o


@tool(
    "find_items_by_effect",
    "按业务目标反查配置项：给定一个业务指标，返回所有能增强它和会削弱它的配置项。"
    "做推荐类任务、需要围绕某个目标选玩法时使用。",
    {
        "effect": {"type": "string", "description": "业务指标：author_income / reader_retention / "
                                                    "newbook_exposure / platform_revenue / update_stability，"
                                                    "也可以直接用中文名"},
    },
)
def find_items_by_effect(effect: str) -> tuple[str, dict]:
    reg = get_registry()
    key = effect
    if effect not in reg.effects:
        key = next((k for k, v in reg.effects.items() if effect in (k, v)), None)
    if key is None:
        return f"未知业务指标 {effect}。可选：{'、'.join(reg.effects)}", {}
    pos, neg = reg.items_affecting(key)
    lines = [f"对「{reg.effects[key]}」的影响："]
    lines.append("增强：" + "、".join(f"{i.name}({i.key})" for i in pos) if pos else "增强：无")
    lines.append("削弱：" + "、".join(f"{i.name}({i.key})" for i in neg) if neg else "削弱：无")
    return "\n".join(lines), {
        "enhances": [i.key for i in pos], "weakens": [i.key for i in neg]
    }


@tool(
    "auto_fix_plan",
    "对一套存在依赖缺失的配置方案自动补齐依赖项。注意它只修「答案唯一」的问题，"
    "互斥冲突和分成上限这类需要业务取舍的，它不会替你决定。",
    {"values": {"type": "object", "description": "配置项 key -> 取值"}},
)
def auto_fix_plan(values: dict) -> tuple[str, dict]:
    engine = get_rule_engine()
    fixed, notes = engine.auto_fix(ConfigPlan(values=dict(values or {})))
    report = engine.validate(fixed)
    lines = ["【自动补齐】" + ("\n".join(notes) if notes else "无需补齐")]
    lines.append("\n【补齐后校验】\n" + report.to_text())
    if not report.passed:
        lines.append("\n仍存在需要业务取舍的冲突，请在回答中把选项和代价讲清楚，由编辑决定。")
    return "\n".join(lines), {
        "values": fixed.values, "notes": notes, "passed": report.passed
    }


# =============================================================================
# 调度
# =============================================================================


def tool_specs() -> list[dict]:
    return [
        {"name": name, "description": spec["description"], "parameters": spec["parameters"]}
        for name, spec in TOOLS.items()
    ]


def tools_prompt() -> str:
    lines: list[str] = []
    for name, spec in TOOLS.items():
        params = ", ".join(
            f"{k}: {v['type']} — {v['description']}" for k, v in spec["parameters"].items()
        )
        lines.append(f"- {name}({params})\n  {spec['description']}")
    return "\n".join(lines)


def run_tool(name: str, arguments: dict) -> ToolResult:
    t0 = time.perf_counter()
    spec = TOOLS.get(name)
    if spec is None:
        return ToolResult(
            name=name, ok=False,
            content=f"不存在名为 {name} 的工具。可用工具：{'、'.join(TOOLS)}",
        )
    try:
        content, data = spec["fn"](**(arguments or {}))
        ok = True
    except TypeError as e:
        content, data, ok = f"参数错误：{e}。请检查参数名和类型后重试。", {}, False
    except Exception as e:  # noqa: BLE001 - 工具异常要变成可读反馈回灌给 LLM，而不是炸掉整轮
        content, data, ok = f"工具执行失败：{type(e).__name__}: {e}", {}, False
    return ToolResult(
        name=name, ok=ok, content=content, data=data,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
