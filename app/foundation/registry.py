"""配置注册表加载器。

config_registry.yaml 模拟的是配置平台元数据库的导出，规则引擎、Agent 工具、
模拟配置存储都从这里读同一份定义，避免出现「规则引擎认为 T1=50%、
文案里写 T1=48%」这种不一致。

注意：**在线检索链路不读这个文件**。图谱是从 raw_docs 里抽出来的，
读注册表建图等于把答案直接喂给系统，评测就没有意义了。
这里只服务于「确定性校验」和「取值合法性」这两件必须精确的事。
"""
from __future__ import annotations

import functools
from typing import Any

import yaml

from app.foundation.settings import settings


class ConfigItemDef:
    def __init__(self, raw: dict) -> None:
        self.key: str = raw["key"]
        self.name: str = raw["name"]
        self.aliases: list[str] = list(raw.get("aliases", []))
        self.category: str = raw.get("category", "")
        self.dimensions: list[str] = list(raw.get("dimensions", []))
        self.type: str = raw.get("type", "bool")
        self.options: list[Any] = list(raw.get("options", []))
        self.range: list[float] | None = raw.get("range")
        self.default: Any = raw.get("default")
        self.unit: str = raw.get("unit", "")
        self.desc: str = raw.get("desc", "")
        self.depends_on: list[str] = list(raw.get("depends_on", []))
        self.requires_value: dict[str, Any] = dict(raw.get("requires_value", {}))
        self.mutex_with: list[str] = list(raw.get("mutex_with", []))
        self.enhances: list[str] = list(raw.get("enhances", []))
        self.weakens: list[str] = list(raw.get("weakens", []))

    def is_enabled(self, value: Any) -> bool:
        """判断某个取值是否算「开启」—— 依赖与互斥判定都建立在这个语义上。

        bool 看真假；数值型看是否 > 0（0 在这套配置里统一表示「不生效 / 不限制」）；
        枚举型看是否为 off。这个约定在文档里是隐含的，这里显式化。
        """
        if value is None:
            return False
        if self.type == "bool":
            return bool(value)
        if self.type in ("int", "float", "percent"):
            try:
                return float(value) > 0
            except (TypeError, ValueError):
                return False
        return str(value).lower() not in ("off", "none", "false", "")

    def validate_value(self, value: Any) -> str | None:
        """返回错误信息，合法则返回 None。"""
        if self.type == "bool":
            if not isinstance(value, bool):
                return f"{self.name} 应为布尔值，收到 {value!r}"
            return None
        if self.type == "enum":
            if value not in self.options:
                return f"{self.name} 取值 {value!r} 不在允许范围 {self.options} 内"
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return f"{self.name} 应为数值，收到 {value!r}"
        if self.range and not (self.range[0] <= num <= self.range[1]):
            return f"{self.name} 取值 {value} 超出允许范围 [{self.range[0]}, {self.range[1]}]"
        return None

    def describe(self) -> str:
        bits = [f"{self.name}（{self.key}）", f"类型：{self.type}"]
        if self.options:
            bits.append(f"可选值：{'/'.join(map(str, self.options))}")
        if self.range:
            bits.append(f"范围：{self.range[0]}~{self.range[1]}{self.unit}")
        bits.append(f"默认：{self.default}")
        bits.append(f"维度：{'/'.join(self.dimensions)}")
        return " | ".join(bits) + f"\n{self.desc}"


class Registry:
    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.items: dict[str, ConfigItemDef] = {
            i["key"]: ConfigItemDef(i) for i in raw.get("items", [])
        }
        self.effects: dict[str, str] = {e["key"]: e["name"] for e in raw.get("effects", [])}
        self.categories: dict[str, str] = {
            c["key"]: c["name"] for c in raw.get("categories", [])
        }
        self.dimensions: list[dict] = raw.get("meta", {}).get("dimensions", [])
        self.priority_order: list[str] = raw.get("meta", {}).get("priority_order", [])
        self.composite_rules: list[dict] = raw.get("composite_rules", [])
        self.commission_pct: dict[str, int] = raw.get("commission_pct_map", {})

        # name/alias -> key 的反查表，Agent 工具需要接受中文名入参
        self.name_to_key: dict[str, str] = {}
        for key, item in self.items.items():
            for surface in [key, item.name, *item.aliases]:
                self.name_to_key[surface.lower()] = key

    def resolve(self, name_or_key: str) -> ConfigItemDef | None:
        key = self.name_to_key.get(str(name_or_key).strip().lower())
        return self.items.get(key) if key else None

    def dimension_names(self) -> dict[str, str]:
        return {d["key"]: d["name"] for d in self.dimensions}

    def items_by_category(self, category: str) -> list[ConfigItemDef]:
        return [i for i in self.items.values() if i.category == category]

    def items_affecting(self, effect_key: str) -> tuple[list[ConfigItemDef], list[ConfigItemDef]]:
        """返回 (增强该效果的配置项, 削弱该效果的配置项)。"""
        pos = [i for i in self.items.values() if effect_key in i.enhances]
        neg = [i for i in self.items.values() if effect_key in i.weakens]
        return pos, neg

    def dependency_chain(self, key: str, _seen: set[str] | None = None) -> list[str]:
        """向上追溯完整依赖链，去重且防环。"""
        seen = _seen if _seen is not None else set()
        item = self.items.get(key)
        if item is None or key in seen:
            return []
        seen.add(key)
        chain: list[str] = []
        for dep in [*item.depends_on, *item.requires_value.keys()]:
            if dep in chain or dep in seen:
                continue
            chain.extend(self.dependency_chain(dep, seen))
            if dep not in chain:
                chain.append(dep)
        return chain


@functools.lru_cache(maxsize=1)
def get_registry() -> Registry:
    path = settings.gold_dir / "config_registry.yaml"
    return Registry(yaml.safe_load(path.read_text(encoding="utf-8")))
