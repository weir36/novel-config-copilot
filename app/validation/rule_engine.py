"""规则校验引擎 —— 系统里唯一的确定性组件。

它存在的理由很直接：LLM 生成的配置方案可以有创意，但不能违法。
依赖缺失、互斥冲突、分成上限这些约束是布尔判定，用大模型做既贵又不稳，
用规则做既免费又 100% 准。所以这里的分工是：
**LLM 负责"配什么"，规则引擎负责"能不能配"。**

它在系统里出现两次：
  1. 作为 Agent 的工具（check_constraints）—— 让 Agent 在生成过程中自己发现冲突并改，
     而不是生成完了被外面拦下来重来一遍。
  2. 作为生成后的最终兜底 —— Agent 可能在 max_steps 内没收敛，最后必须再过一道。

复合规则 R001–R004 用 Python 函数实现而不是 eval(YAML 里的表达式)：
表达式串留在 YAML 里当文档，执行走显式代码。规则引擎是安全边界，不该有 eval。
"""
from __future__ import annotations

from typing import Any, Callable

from app.foundation.registry import ConfigItemDef, Registry, get_registry
from app.foundation.models import (
    ConfigPlan,
    ValidationReport,
    Violation,
    ViolationSeverity,
)

CompositeFn = Callable[[dict[str, Any], Registry], Violation | None]
_COMPOSITE: dict[str, CompositeFn] = {}


def composite(rule_id: str):
    def deco(fn: CompositeFn) -> CompositeFn:
        _COMPOSITE[rule_id] = fn
        return fn

    return deco


def _eff(values: dict[str, Any], key: str, reg: Registry) -> Any:
    """取生效值：方案里显式配了就用配的，没配就用注册表默认值。

    这一步很关键。编辑提交的方案往往只包含"本次要改的几项"，
    但复合规则涉及的其他配置项此刻是有当前值的。只看方案里有的字段做校验，
    会漏掉绝大多数真实冲突 —— 线上错误配置里 80% 恰恰是
    "新配的项 × 没动的老项"组合违规。
    """
    if key in values:
        return values[key]
    item = reg.items.get(key)
    return item.default if item else None


# =============================================================================
# 复合规则
# =============================================================================


@composite("R001")
def _r001_buyout_no_completion(values: dict, reg: Registry) -> Violation | None:
    if _eff(values, "contract_type", reg) != "buyout":
        return None
    if not _eff(values, "completion_bonus", reg):
        return None
    return Violation(
        rule_id="R001",
        severity=ViolationSeverity.ERROR,
        message="签约类型为买断时不能开启完本奖（买断稿费已一次性付清，不再额外奖励）",
        involved_items=["contract_type", "completion_bonus"],
        suggestion="关闭完本奖；如果作者确实需要完本激励，下一本书签成分成或保底分成",
    )


@composite("R002")
def _r002_attendance_words(values: dict, reg: Registry) -> Violation | None:
    if not _eff(values, "full_attendance_bonus", reg):
        return None
    words = _eff(values, "daily_update_words", reg) or 0
    # 日更字数为 0 时属于依赖缺失，由依赖校验报，这里不重复报
    if float(words) <= 0 or float(words) >= 4000:
        return None
    return Violation(
        rule_id="R002",
        severity=ViolationSeverity.ERROR,
        message=f"开启全勤奖时日更字数要求不得低于 4000 字，当前为 {words} 字",
        involved_items=["full_attendance_bonus", "daily_update_words"],
        suggestion="把日更字数要求调到 4000 或以上，或者先不开全勤奖",
    )


@composite("R003")
def _r003_guarantee_share_cap(values: dict, reg: Registry) -> Violation | None:
    if _eff(values, "contract_type", reg) != "guarantee":
        return None
    ratio = _eff(values, "share_ratio", reg)
    if ratio is None or float(ratio) <= 50:
        return None
    return Violation(
        rule_id="R003",
        severity=ViolationSeverity.ERROR,
        message=f"保底分成作品的作者分成比例不得超过 50%，当前为 {ratio}%",
        involved_items=["contract_type", "share_ratio"],
        suggestion="把作者分成比例降到 50% 以内；想给作者更多收入，可以提高保底金额",
    )


@composite("R004")
def _r004_early_paywall(values: dict, reg: Registry) -> Violation | None:
    if not _eff(values, "paid_chapter", reg):
        return None
    start = _eff(values, "paywall_start_chapter", reg) or 0
    if not (0 < float(start) < 20):
        return None
    return Violation(
        rule_id="R004",
        severity=ViolationSeverity.WARN,
        message=f"付费起始章节为第 {start} 章，早于第 20 章，读者留存可能明显下滑",
        involved_items=["paid_chapter", "paywall_start_chapter"],
        suggestion="把付费起始章节往后调到第 20 章以后，或先和作者确认前期剧情能否留住读者",
    )


# =============================================================================
# 引擎
# =============================================================================


class RuleEngine:
    def __init__(self, registry: Registry | None = None) -> None:
        self.reg = registry or get_registry()

    def validate(self, plan: ConfigPlan, strict_dimension: bool = True) -> ValidationReport:
        values = dict(plan.values)
        violations: list[Violation] = []

        violations.extend(self._check_known_and_types(values))
        if strict_dimension:
            violations.extend(self._check_dimensions(plan))
        violations.extend(self._check_dependencies(values))
        violations.extend(self._check_mutex(values))
        violations.extend(self._check_composite(values))

        # 同一 rule_id + involved 的重复项去掉，避免互斥双向各报一次
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped: list[Violation] = []
        for v in violations:
            k = (v.rule_id, tuple(sorted(v.involved_items)))
            if k in seen:
                continue
            seen.add(k)
            deduped.append(v)

        has_error = any(v.severity == ViolationSeverity.ERROR for v in deduped)
        return ValidationReport(passed=not has_error, violations=deduped)

    # ---------------- 各类校验 ----------------
    def _check_known_and_types(self, values: dict) -> list[Violation]:
        out: list[Violation] = []
        for key, value in values.items():
            item = self.reg.items.get(key)
            if item is None:
                out.append(
                    Violation(
                        rule_id="UNKNOWN_ITEM",
                        severity=ViolationSeverity.ERROR,
                        message=f"配置项 {key} 不存在于配置平台",
                        involved_items=[key],
                        suggestion="检查拼写，或确认该配置项是否已下线",
                    )
                )
                continue
            err = item.validate_value(value)
            if err:
                out.append(
                    Violation(
                        rule_id="INVALID_VALUE",
                        severity=ViolationSeverity.ERROR,
                        message=err,
                        involved_items=[key],
                    )
                )
        return out

    def _check_dimensions(self, plan: ConfigPlan) -> list[Violation]:
        out: list[Violation] = []
        names = self.reg.dimension_names()
        for key, dim in plan.dimensions.items():
            item = self.reg.items.get(key)
            if item is None or not dim:
                continue
            if dim not in item.dimensions:
                out.append(
                    Violation(
                        rule_id="INVALID_DIMENSION",
                        severity=ViolationSeverity.ERROR,
                        message=(f"{item.name} 不支持在「{names.get(dim, dim)}」维度配置，"
                                 f"仅支持 {'/'.join(names.get(d, d) for d in item.dimensions)}"),
                        involved_items=[key],
                        suggestion=f"改到 {names.get(item.dimensions[0], item.dimensions[0])} 维度配置",
                    )
                )
        return out

    def _check_dependencies(self, values: dict) -> list[Violation]:
        out: list[Violation] = []
        for key, value in values.items():
            item = self.reg.items.get(key)
            if item is None or not item.is_enabled(value):
                continue  # 没开启的配置项不触发依赖校验

            for dep_key in item.depends_on:
                dep = self.reg.items.get(dep_key)
                if dep is None:
                    continue
                dep_val = _eff(values, dep_key, self.reg)
                if not dep.is_enabled(dep_val):
                    chain = self.reg.dependency_chain(key)
                    chain_txt = " → ".join(
                        self.reg.items[k].name for k in [*chain, key] if k in self.reg.items
                    )
                    out.append(
                        Violation(
                            rule_id="DEPENDENCY_MISSING",
                            severity=ViolationSeverity.ERROR,
                            message=(f"{item.name} 依赖 {dep.name}，但后者当前为"
                                     f"「{dep_val}」（未开启）"),
                            involved_items=[key, dep_key],
                            suggestion=f"先开启 {dep.name}。完整依赖链：{chain_txt}",
                        )
                    )

            for dep_key, required in item.requires_value.items():
                dep = self.reg.items.get(dep_key)
                if dep is None:
                    continue
                actual = _eff(values, dep_key, self.reg)
                if not _value_satisfies(actual, required):
                    out.append(
                        Violation(
                            rule_id="REQUIRED_VALUE_UNMET",
                            severity=ViolationSeverity.ERROR,
                            message=(f"{item.name} 要求 {dep.name} = {required}，"
                                     f"当前为 {actual}"),
                            involved_items=[key, dep_key],
                            suggestion=f"将 {dep.name} 调整为 {required}",
                        )
                    )
        return out

    def _check_mutex(self, values: dict) -> list[Violation]:
        out: list[Violation] = []
        enabled = {
            k for k, v in values.items()
            if k in self.reg.items and self.reg.items[k].is_enabled(v)
        }
        for key in list(enabled):
            item = self.reg.items[key]
            for other_key in item.mutex_with:
                other = self.reg.items.get(other_key)
                if other is None:
                    continue
                other_val = _eff(values, other_key, self.reg)
                if not other.is_enabled(other_val):
                    continue
                out.append(
                    Violation(
                        rule_id="MUTEX_CONFLICT",
                        severity=ViolationSeverity.ERROR,
                        message=f"{item.name} 与 {other.name} 互斥，不能同时开启",
                        involved_items=sorted([key, other_key]),
                        suggestion=f"二者择一：保留 {item.name} 则需关闭 {other.name}",
                    )
                )
        return out

    def _check_composite(self, values: dict) -> list[Violation]:
        out: list[Violation] = []
        for rule in self.reg.composite_rules:
            fn = _COMPOSITE.get(rule["id"])
            if fn is None:
                continue
            v = fn(values, self.reg)
            if v is not None:
                out.append(v)
        return out

    # ---------------- 自动修复 ----------------
    def auto_fix(self, plan: ConfigPlan, max_rounds: int = 3) -> tuple[ConfigPlan, list[str]]:
        """对可确定性修复的违规自动改写方案，返回 (修复后方案, 修复说明)。

        只修「答案唯一」的违规：依赖缺失就把依赖打开、requires_value 不满足就改成要求值。
        互斥和分成上限这类**需要业务取舍**的冲突不自动改 —— 关哪个、降哪边是业务决策，
        系统替用户做这个决定是危险的，正确做法是把选项和代价讲清楚让人来选。
        """
        fixed = plan.model_copy(deep=True)
        notes: list[str] = []
        for _ in range(max_rounds):
            report = self.validate(fixed)
            actionable = [
                v for v in report.violations
                if v.rule_id in ("DEPENDENCY_MISSING", "REQUIRED_VALUE_UNMET")
            ]
            if not actionable:
                break
            for v in actionable:
                target = v.involved_items[1] if len(v.involved_items) > 1 else None
                item = self.reg.items.get(target) if target else None
                if item is None:
                    continue
                if v.rule_id == "DEPENDENCY_MISSING":
                    new_val = True if item.type == "bool" else _min_enabled_value(item)
                else:
                    owner = self.reg.items.get(v.involved_items[0])
                    new_val = owner.requires_value.get(target) if owner else None
                    if isinstance(new_val, str) and new_val.endswith("+"):
                        new_val = new_val[:-1]  # "v3.2+" -> "v3.2"
                if new_val is None:
                    continue
                fixed.values[target] = new_val
                notes.append(f"自动补齐：{item.name} → {new_val}（原因：{v.message}）")
        return fixed, notes


def _value_satisfies(actual: Any, required: Any) -> bool:
    if isinstance(required, str) and required.endswith("+"):
        # 版本号下限语义，如 "v3.2+"
        return _version_ge(str(actual), required[:-1])
    if isinstance(required, bool):
        return bool(actual) == required
    return str(actual) == str(required)


def _version_ge(a: str, b: str) -> bool:
    def parts(v: str) -> list[int]:
        return [int(x) for x in v.lstrip("vV").split(".") if x.isdigit()]

    try:
        return parts(a) >= parts(b)
    except ValueError:
        return False


def _min_enabled_value(item: ConfigItemDef) -> Any:
    if item.type == "bool":
        return True
    if item.type == "enum":
        return next((o for o in item.options if str(o).lower() != "off"), item.options[0] if item.options else None)
    lo = item.range[0] if item.range else 0
    return max(lo, 1)


_ENGINE: RuleEngine | None = None


def get_rule_engine() -> RuleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RuleEngine()
    return _ENGINE
