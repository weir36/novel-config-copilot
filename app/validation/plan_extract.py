"""从自然语言回答中抽取结构化配置方案。

用途是给规则引擎喂数据 —— LLM 输出的是给人读的 markdown，规则引擎要的是 dict。

**优先走规则抽取**而不是再调一次 LLM，原因有三：
  1. 这一步在每次产出方案时都跑，多一次 LLM 调用就是多一份延迟和成本；
  2. 我们的提示词已经要求输出 markdown 表格，格式是可预期的；
  3. 规则抽取的失败是"漏抽"（安全），LLM 抽取的失败可能是"抽错值"（危险）——
     校验一个被抽错的值，结论是错的，比没校验更糟。

规则抽不出东西时才回落到 LLM，并且 LLM 抽完的结果会再过一遍取值合法性校验。
"""
from __future__ import annotations

import logging
import re

from app.foundation.registry import get_registry
from app.generation.llm_client import LLMError, chat_json
from app.generation.prompts import PLAN_EXTRACT_PROMPT
from app.foundation.models import ConfigPlan

logger = logging.getLogger(__name__)

DIM_ALIAS = {
    "频道": "channel", "channel": "channel", "男频": "channel", "女频": "channel",
    "作者": "author", "author": "author",
    "作品": "book", "书": "book", "book": "book",
}

# 枚举值的中文写法。长的放前面："保底分成"要先于"分成"匹配
ENUM_ALIAS = [("保底分成", "guarantee"), ("买断", "buyout"), ("分成", "share")]

TRUE_WORDS = {"true", "开启", "开", "是", "yes", "on", "启用", "打开"}
FALSE_WORDS = {"false", "关闭", "关", "否", "no", "off", "禁用", "不开"}

_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def extract_plan(answer: str, use_llm_fallback: bool = True) -> ConfigPlan:
    plan = _extract_by_rule(answer)
    if plan.values:
        return plan
    if not use_llm_fallback:
        return plan
    try:
        return _extract_by_llm(answer)
    except LLMError as e:
        logger.warning("LLM 方案抽取失败：%s", e)
        return plan


# =============================================================================
# 规则抽取
# =============================================================================


# 表头关键词 -> 列语义。按优先级排列：出现"建议值"就以它为准，忽略"当前值"。
_VALUE_HEADERS = ("建议值", "目标值", "调整为", "新值", "取值", "配置值", "值")
_DIM_HEADERS = ("维度", "生效维度", "配置维度")
_WHY_HEADERS = ("理由", "原因", "说明", "备注")


def _extract_by_rule(answer: str) -> ConfigPlan:
    reg = get_registry()
    plan = ConfigPlan()
    # 一次回答里可能有多张表（配置清单 + 变更项），每张表的列结构不同，
    # 所以列索引要跟着表头走，遇到新表头就重置。
    cols: dict[str, int] | None = None

    for line in answer.splitlines():
        m = _TABLE_ROW.match(line)
        if not m:
            # 表结束，下一张表要重新识别表头
            if line.strip() and not line.strip().startswith("|"):
                cols = None
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 2 or set("".join(cells)) <= set("-: "):
            continue  # 分隔行

        header = _parse_header(cells)
        if header is not None:
            cols = header
            continue

        item = _resolve_cell(cells[0], reg)
        if item is None:
            continue

        # 早先这里用「四列且二三列都非空」来猜是不是变更表，
        # 结果把「配置项 | 取值 | 维度 | 理由」误判成「当前值 | 建议值」，
        # 把维度列当成了取值。读表头定位列才是可靠做法。
        vi = (cols or {}).get("value", 1)
        raw_value = cells[vi] if vi < len(cells) else cells[1]
        value = _coerce(raw_value, item)
        if value is None:
            continue
        plan.values[item.key] = value

        di = (cols or {}).get("dim")
        if di is not None and di < len(cells):
            dim = _resolve_dim(cells[di])
            if dim:
                plan.dimensions[item.key] = dim
        else:
            for cell in cells[1:]:
                dim = _resolve_dim(cell)
                if dim:
                    plan.dimensions[item.key] = dim
                    break

        wi = (cols or {}).get("why")
        if wi is not None and wi < len(cells):
            plan.rationale[item.key] = cells[wi][:120]
        elif len(cells) >= 3:
            plan.rationale[item.key] = cells[-1][:120]

    return plan


def _parse_header(cells: list[str]) -> dict[str, int] | None:
    """识别表头行并定位各列。不是表头就返回 None。"""
    first = cells[0]
    if not any(k in first for k in ("配置项", "项目", "配置")):
        return None
    cols: dict[str, int] = {}
    for i, cell in enumerate(cells[1:], start=1):
        if "value" not in cols and any(h in cell for h in _VALUE_HEADERS):
            # "当前值" 不能当目标值；只有后面没有更合适的列时才退而用它
            if "当前" in cell:
                cols.setdefault("_current", i)
                continue
            cols["value"] = i
        elif "dim" not in cols and any(h in cell for h in _DIM_HEADERS):
            cols["dim"] = i
        elif "why" not in cols and any(h in cell for h in _WHY_HEADERS):
            cols["why"] = i
    cols.setdefault("value", cols.pop("_current", 1))
    cols.pop("_current", None)
    return cols


def _resolve_cell(cell: str, reg) -> object | None:
    # 单元格常见形态："全勤奖（full_attendance_bonus）"、"`full_attendance_bonus`"、"全勤奖"
    cell = cell.strip().strip("`*")
    m = re.search(r"[（(]([a-z0-9_]+)[）)]", cell)
    if m:
        item = reg.resolve(m.group(1))
        if item:
            return item
    m = re.fullmatch(r"[a-z][a-z0-9_]*", cell)
    if m:
        return reg.resolve(cell)
    cleaned = re.sub(r"[（(].*?[）)]", "", cell).strip()
    return reg.resolve(cleaned) or reg.resolve(cell)


def _resolve_dim(cell: str) -> str | None:
    c = cell.strip().lower()
    if c in DIM_ALIAS:
        return DIM_ALIAS[c]
    # 维度写在取值的括号注释里："True（作品维度）"
    for note in _NOTE_RE.findall(c):
        note = note.strip()
        if note in DIM_ALIAS:
            return DIM_ALIAS[note]
        if note.endswith("维度") and note[:-2] in DIM_ALIAS:
            return DIM_ALIAS[note[:-2]]
    return None


_NOTE_RE = re.compile(r"[（(]([^）)]*)[）)]")


def _coerce(raw: str, item) -> object | None:
    raw = raw.strip().strip("`*").replace("％", "%")
    # 去掉括号注释："True（作品维度）"、"6000（字/日）"。不去掉的话布尔值匹配不上，
    # 整行被跳过，一套合法方案就被当成"没抽到方案"
    bare = _NOTE_RE.sub("", raw).strip().strip("`*") or raw
    if not raw or bare in ("-", "—", "不变", "维持"):
        return None
    raw = bare

    if item.type == "bool":
        low = raw.lower()
        if low in TRUE_WORDS or "开启" in raw:
            return True
        if low in FALSE_WORDS or "关闭" in raw:
            return False
        return None

    if item.type == "enum":
        for opt in item.options:
            if str(opt).lower() == raw.lower():
                return opt
        # 允许 "guarantee（保底分成）" 这种带注释的写法
        for opt in item.options:
            if re.search(rf"\b{re.escape(str(opt))}\b", raw, re.IGNORECASE):
                return opt
        for zh, opt in ENUM_ALIAS:
            if zh in raw and opt in item.options:
                return opt
        return None

    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return None
    num = float(m.group())
    return int(num) if item.type == "int" or num.is_integer() else num


# =============================================================================
# LLM 兜底
# =============================================================================


def _extract_by_llm(answer: str) -> ConfigPlan:
    reg = get_registry()
    item_desc = "\n".join(
        f"- {k}（{i.name}）类型={i.type}"
        + (f" 可选={i.options}" if i.options else "")
        + (f" 范围={i.range}" if i.range else "")
        for k, i in reg.items.items()
    )
    prompt = PLAN_EXTRACT_PROMPT.format(item_keys=item_desc, answer=answer[:6000])
    data = chat_json([{"role": "user", "content": prompt}], purpose="plan", max_tokens=1500)
    if not isinstance(data, dict):
        return ConfigPlan()

    plan = ConfigPlan()
    for key, value in (data.get("values") or {}).items():
        item = reg.items.get(key)
        if item is None:
            continue
        # LLM 抽出来的值必须过一遍合法性，抽错的值比没抽到更危险
        if item.validate_value(value) is None:
            plan.values[key] = value
    for key, dim in (data.get("dimensions") or {}).items():
        if key in plan.values and dim in DIM_ALIAS.values():
            plan.dimensions[key] = dim
    for key, why in (data.get("rationale") or {}).items():
        if key in plan.values:
            plan.rationale[key] = str(why)[:120]
    return plan
