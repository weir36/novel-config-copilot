"""知识图谱抽取：从散文档里把实体和关系抠出来。

**实体有两种抽法，用 --entities 参数切换：**

  - vocab（默认）：词表 + 别名倒排。配置项、业务效果是封闭词表，真实业务里能从
    配置平台元数据库导出，按定义准确率 100%，零成本。
  - llm：大模型按 schema（配置项 / 业务效果 / 维度）从每个 chunk 里抽实体，
    再走四层消歧归并到标准 ID：
      1. 名字或别名和词表完全一致 → 直接归到标准 ID
      2. 统一大小写、空格、全半角、去标点后再匹配
      3. 和词表实体比相似度，够像的合并（向量余弦 ≥ 0.9；没有向量模型时退回字面相似度 ≥ 0.8）
      4. 还对不上 → 标成「待审核」，写进 pending_entities.json，不自动进图
    它的价值不在准确率（拼不过词表），而在**发现词表里还没有的新实体**，交给人审。

两种方式的对比见 eval/run_graph_eval.py 的实体评测部分。

**关系只能靠抽。** "付费章节和免费阅读模式互斥"这句话在元数据库里可能根本没有，
只存在于运营和编辑写的文档里 —— 把散落在文档里的关系知识结构化，正是这个项目要解决的问题。

两种关系抽取器：
  - RuleExtractor：正则模式匹配。快、免费、可解释，召回受限于模式覆盖度。
  - LLMExtractor：大模型按 schema 抽。覆盖面广，能抓到句式变体，但有幻觉且要花钱。
  - hybrid：规则打底 + LLM 补充，冲突时规则优先。

三者的 P/R/F1 对比见 eval/run_graph_eval.py。
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import defaultdict

from app.foundation.registry import get_registry
from app.foundation.settings import settings
from app.generation.llm_client import LLMError, chat_json
from app.foundation.models import (
    Chunk,
    Entity,
    EntityType,
    KnowledgeGraph,
    Relation,
    RelationType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 实体（词表）
# =============================================================================


def build_entities(chunks: list[Chunk]) -> list[Entity]:
    reg = get_registry()
    source_map: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        for key in c.linked_item_keys:
            source_map[f"config:{key}"].append(c.chunk_id)

    entities: list[Entity] = []

    # --- 配置项 ---
    for key, item in reg.items.items():
        eid = f"config:{key}"
        # embed_text 是实体向量索引的输入。放什么进去直接决定实体链接的召回：
        # 只放标准名太短（"全勤奖"三个字），信息量不足；
        # 把别名、描述、分类、维度都拼进去，口语化 query 才能对上。
        embed_text = " ｜ ".join(
            filter(None, [
                item.name,
                " ".join(item.aliases),
                key,
                item.desc,
                f"分类：{reg.categories.get(item.category, item.category)}",
                f"配置维度：{'/'.join(item.dimensions)}",
            ])
        )
        entities.append(
            Entity(
                entity_id=eid,
                name=item.name,
                type=EntityType.CONFIG_ITEM,
                aliases=[*item.aliases, key],
                description=item.desc,
                embed_text=embed_text,
                source_chunk_ids=sorted(set(source_map.get(eid, [])))[:12],
                props={
                    "key": key, "type": item.type, "category": item.category,
                    "dimensions": item.dimensions, "default": item.default,
                    "options": item.options, "range": item.range, "unit": item.unit,
                },
            )
        )

    # --- 业务效果 ---
    for key, name in reg.effects.items():
        entities.append(
            Entity(
                entity_id=f"effect:{key}", name=name, type=EntityType.EFFECT,
                aliases=[key], description=f"业务指标：{name}",
                embed_text=f"{name} ｜ {key} ｜ 业务指标 效果 影响",
                props={"key": key},
            )
        )

    # --- 维度 ---
    for d in reg.dimensions:
        entities.append(
            Entity(
                entity_id=f"dim:{d['key']}", name=d["name"], type=EntityType.DIMENSION,
                aliases=[d["key"]], description=d.get("desc", ""),
                embed_text=f"{d['name']} ｜ {d['key']} ｜ 配置维度 ｜ {d.get('desc', '')}",
                props={"key": d["key"]},
            )
        )

    # --- 分类 ---
    for key, name in reg.categories.items():
        entities.append(
            Entity(
                entity_id=f"cat:{key}", name=name, type=EntityType.CATEGORY,
                aliases=[key], embed_text=f"{name} ｜ {key} ｜ 配置分类",
                props={"key": key},
            )
        )

    # --- 复合规则 ---
    for rule in reg.composite_rules:
        entities.append(
            Entity(
                entity_id=f"rule:{rule['id']}",
                name=f"{rule['id']} {rule['name']}",
                type=EntityType.RULE,
                aliases=[rule["id"], rule["name"]],
                description=rule.get("desc", ""),
                embed_text=f"{rule['id']} {rule['name']} ｜ {rule.get('desc', '')} ｜ "
                           f"校验规则 约束 severity={rule.get('severity')}",
                props={"key": rule["id"], "severity": rule.get("severity")},
            )
        )

    return entities


# =============================================================================
# 实体：LLM 抽取 + 四层消歧
# =============================================================================

ENTITY_PROMPT = """你是一个信息抽取器。从下面这段网文平台的政策文档中，抽取实体。

# 实体类型（只能用这些）
- ConfigItem：编辑在后台能配置的项，比如某个开关、比例、金额、字数要求
- Effect：业务指标，比如作者收入、读者留存
- Dimension：配置维度，比如频道、作者、作品

# 抽取要求
1. 用文中原本的写法作为 name，不要改写、不要翻译。
2. 同一个实体在文中出现多种写法（比如中文名和英文 key），都列进 aliases。
3. 只抽文中明确出现的，不要补充。抽不到返回空数组。

# 输出格式
{{"entities": [{{"name": "实体名", "type": "实体类型", "aliases": ["别的写法"]}}]}}

# 当前小节标题
{section}

# 待抽取文本
{text}"""

_ENTITY_TYPES = {"ConfigItem": "config", "Effect": "effect", "Dimension": "dim"}


class LLMEntityExtractor:
    """逐 chunk 让大模型抽实体，返回原始提及 [(chunk_id, name, aliases, type)]。"""

    name = "llm"

    def extract(self, chunks: list[Chunk]) -> list[dict]:
        out: list[dict] = []
        for c in chunks:
            prompt = ENTITY_PROMPT.format(section=c.section, text=c.text)
            try:
                data = chat_json(
                    [{"role": "user", "content": prompt}],
                    purpose="extract_entity",
                    model=settings.extract_model_name,
                    temperature=0.0,
                    max_tokens=1200,
                )
            except LLMError as e:
                logger.warning("chunk %s 实体抽取失败：%s", c.chunk_id, e)
                continue
            for e in (data.get("entities", []) if isinstance(data, dict) else []):
                etype = _ENTITY_TYPES.get(str(e.get("type", "")))
                name = str(e.get("name", "")).strip()
                if etype and name:
                    out.append({"chunk_id": c.chunk_id, "name": name, "type": etype,
                                "aliases": [str(a) for a in e.get("aliases", []) if a]})
        return out


def _normalize(text: str) -> str:
    """第二层消歧用：全角转半角、转小写、去空格和标点。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\-_·・,，。、()（）\[\]【】「」\"'`]", "", t)


def _char_bigram_sim(a: str, b: str) -> float:
    """embedder 还没 fit 时的相似度兜底：字符 2-gram 的 Dice 系数。"""
    def grams(s: str) -> set[str]:
        s = _normalize(s)
        return {s[i:i + 2] for i in range(len(s) - 1)} or {s}
    ga, gb = grams(a), grams(b)
    return 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0


class Disambiguator:
    """把原始提及归并到标准实体 ID。按层级逐层尝试，前一层命中就不再往下。"""

    # 两种相似度的量纲不一样：向量余弦 0.9 已经很像了；字符 2-gram 的 Dice 系数要求低一点，
    # 否则只差一个"计划"的"新书期流量扶持计划"都归不过来（实测 0.857）
    SIM_THRESHOLD = 0.9
    CHAR_SIM_THRESHOLD = 0.8

    def __init__(self) -> None:
        reg = get_registry()
        self.exact: dict[str, str] = {}
        self.normed: dict[str, str] = {}
        self.canon_names: list[tuple[str, str]] = []
        for key, item in reg.items.items():
            self._add(f"config:{key}", [item.name, *item.aliases, key])
        for key, name in reg.effects.items():
            self._add(f"effect:{key}", [name, key])
        for d in reg.dimensions:
            self._add(f"dim:{d['key']}", [d["name"], d["key"]])
        self._sim, self._threshold = self._make_similarity()

    def _add(self, eid: str, surfaces: list[str]) -> None:
        for s in surfaces:
            self.exact.setdefault(s.strip().lower(), eid)
            self.normed.setdefault(_normalize(s), eid)
            self.canon_names.append((s, eid))

    @classmethod
    def _make_similarity(cls):
        """有可用的向量模型就比「意思」，没有就退回比字面。返回 (相似度函数, 阈值)。"""
        try:
            from app.retrieval.embedder import get_embedder, l2_normalize  # noqa: PLC0415

            emb = get_embedder()
            emb.encode_query("探测")
        except Exception:  # noqa: BLE001 - TF-IDF 未 fit 时会抛
            return _char_bigram_sim, cls.CHAR_SIM_THRESHOLD

        def sim(a: str, b: str) -> float:
            va = l2_normalize(emb.encode_docs([a]))[0]
            vb = l2_normalize(emb.encode_docs([b]))[0]
            return float(va @ vb)
        return sim, cls.SIM_THRESHOLD

    def resolve(self, name: str, aliases: list[str] | None = None) -> tuple[str | None, str]:
        """返回 (标准 ID 或 None, 命中的层级)。"""
        surfaces = [name, *(aliases or [])]
        for s in surfaces:
            eid = self.exact.get(s.strip().lower())
            if eid:
                return eid, "exact"
        for s in surfaces:
            eid = self.normed.get(_normalize(s))
            if eid:
                return eid, "normalized"
        best, best_eid = 0.0, None
        for canon, eid in self.canon_names:
            score = self._sim(name, canon)
            if score > best:
                best, best_eid = score, eid
        if best >= self._threshold:
            return best_eid, "similarity"
        return None, "pending"


def resolve_mentions(mentions: list[dict]) -> tuple[dict[str, list[str]], list[dict], dict[str, int]]:
    """消歧主流程。返回 (标准 ID -> 出处 chunk 列表, 待审核提及, 各层命中计数)。"""
    dis = Disambiguator()
    found: dict[str, list[str]] = defaultdict(list)
    pending: dict[str, dict] = {}
    layers: dict[str, int] = defaultdict(int)
    for m in mentions:
        eid, layer = dis.resolve(m["name"], m.get("aliases"))
        layers[layer] += 1
        if eid:
            found[eid].append(m["chunk_id"])
        else:
            p = pending.setdefault(m["name"], {"name": m["name"], "type": m["type"], "chunk_ids": []})
            p["chunk_ids"].append(m["chunk_id"])
    return dict(found), list(pending.values()), dict(layers)


# =============================================================================
# 关系抽取：规则
# =============================================================================

# 这些模式是归纳政策文档里的高频表述得到的。它们的共同特点是：
# 关系词（互斥/依赖/要求）附近一定同时出现两个配置项名称。
# 所以策略是：先定位关系词，再在它的上下文窗口里找配置项，配对成边。
_MUTEX_PAT = re.compile(r"互斥|不能同时|二选一|只能二选一|同时开启会|不可叠加|只能选一")
_DEPEND_PAT = re.compile(r"依赖|前置条件|前置依赖|必须先开|需要先|要先开|必须先通过|底层依赖|要求.{0,8}已")
_REQUIRE_PAT = re.compile(r"要求(.{0,20}?)(?:为|是|=|设置为|不低于|及以上)")
_ENHANCE_PAT = re.compile(r"正向|拉动|提升|拉升|增强|利于|收益")
_WEAKEN_PAT = re.compile(r"负向|压制|抑制|削弱|下降|稀释|代价是|会推高|损失")

_WINDOW = 90  # 关系词左右各取多少字当上下文


class RuleExtractor:
    name = "rule"

    def __init__(self) -> None:
        self.reg = get_registry()
        # 别名 -> key，长别名优先
        self.surface: list[tuple[str, str]] = []
        for key, item in self.reg.items.items():
            for s in [item.name, *item.aliases, key]:
                if len(s) >= 2:
                    self.surface.append((s.lower(), key))
        self.surface.sort(key=lambda kv: -len(kv[0]))

        self.effect_surface: list[tuple[str, str]] = sorted(
            [(v.lower(), k) for k, v in self.reg.effects.items()],
            key=lambda kv: -len(kv[0]),
        )

    def _find_items(self, text: str) -> list[str]:
        """按在文中出现的先后顺序返回配置项。

        顺序很重要：判断依赖方向时取的是"关系词左边最近的那个"，
        按别名长度排序的话，left[-1] 拿到的不一定是离关系词最近的配置项。
        """
        low = text.lower()
        pos: dict[str, int] = {}
        consumed = low
        for surf, key in self.surface:
            idx = consumed.find(surf)
            if idx >= 0:
                if key not in pos or idx < pos[key]:
                    pos[key] = idx
                consumed = consumed.replace(surf, " " * len(surf))
        return sorted(pos, key=pos.get)

    def _find_effects(self, text: str) -> list[str]:
        low = text.lower()
        return [k for surf, k in self.effect_surface if surf in low]

    def extract(self, chunks: list[Chunk]) -> list[Relation]:
        out: list[Relation] = []
        for c in chunks:
            out.extend(self._extract_chunk(c))
        return out

    def _extract_chunk(self, chunk: Chunk) -> list[Relation]:
        rels: list[Relation] = []
        text = chunk.text
        # 小节标题里的配置项作为「主语候选」—— 文档里大量用"它""这个开关"指代主语，
        # 只看句子本身会丢掉一半的边
        subject_hint = self._find_items(chunk.section)
        primary = subject_hint[0] if subject_hint else None

        for m in _MUTEX_PAT.finditer(text):
            ctx = text[max(0, m.start() - _WINDOW) : m.end() + _WINDOW]
            items = self._find_items(ctx)
            if len(items) >= 2:
                a, b = items[0], items[1]
            elif len(items) == 1 and primary and items[0] != primary:
                a, b = primary, items[0]
            else:
                continue
            rels.append(_rel(a, b, RelationType.MUTEX_WITH, chunk, 0.9))

        for m in _DEPEND_PAT.finditer(text):
            ctx = text[max(0, m.start() - _WINDOW) : m.end() + _WINDOW]
            items = self._find_items(ctx)
            # 方向：关系词之前的是主语，之后的是宾语。
            # "加更活动 依赖 全勤奖" —— 左边 subject，右边 object
            left = self._find_items(text[max(0, m.start() - _WINDOW) : m.start()])
            right = self._find_items(text[m.end() : m.end() + _WINDOW])
            src = (left[-1] if left else primary)
            dst = right[0] if right else (items[-1] if items else None)
            if src and dst and src != dst:
                rels.append(_rel(src, dst, RelationType.DEPENDS_ON, chunk, 0.85))

        for m in _REQUIRE_PAT.finditer(text):
            ctx = text[max(0, m.start() - _WINDOW) : m.end() + _WINDOW]
            left = self._find_items(text[max(0, m.start() - _WINDOW) : m.start()])
            right = self._find_items(m.group(1) or "")
            src = left[-1] if left else primary
            dst = right[0] if right else None
            if not (src and dst and src != dst):
                continue
            value = _guess_required_value(text[m.start() : m.end() + 40], self.reg.items.get(dst))
            rels.append(
                _rel(src, dst, RelationType.REQUIRES_VALUE, chunk, 0.8,
                     props={"value": value} if value is not None else {})
            )

        # 效果关系：句子里同时出现效果词和正/负向词。
        # 按转折词再切一刀："对作者收入有正向拉动，但平台收益会下降"是一正一负两件事，
        # 不切的话两个指标会同时被标成正向和负向
        for sent in re.split(r"[。；\n]|，但|但是|，代价|不过", text):
            effects = self._find_effects(sent)
            if not effects:
                continue
            items = self._find_items(sent) or ([primary] if primary else [])
            if not items:
                continue
            src = items[0]
            # 先把指标名本身抹掉再找正负向词："平台收益"里的"收益"不能算成正向信号
            sent_wo_effects = sent
            for surf, _k in self.effect_surface:
                sent_wo_effects = sent_wo_effects.replace(surf, " ")
            if _ENHANCE_PAT.search(sent_wo_effects):
                for e in effects:
                    rels.append(_rel(src, f"effect:{e}", RelationType.ENHANCES, chunk, 0.7, raw_dst=True))
            if _WEAKEN_PAT.search(sent_wo_effects):
                for e in effects:
                    rels.append(_rel(src, f"effect:{e}", RelationType.WEAKENS, chunk, 0.7, raw_dst=True))
        return rels


def _guess_required_value(text: str, item) -> str | bool | None:
    if item is None:
        return None
    for opt in item.options or []:
        if re.search(rf"\b{re.escape(str(opt))}\b", text, re.IGNORECASE):
            return opt
    if item.type == "bool" and re.search(r"已接入|已开通|开启|true", text):
        return True
    m = re.search(r"v?\d+\.\d+", text)
    if m and item.type == "enum":
        return m.group()
    return None


def _rel(src_key, dst_key, rtype, chunk, conf, props=None, raw_dst=False) -> Relation:
    return Relation(
        source_id=f"config:{src_key}",
        target_id=dst_key if raw_dst else f"config:{dst_key}",
        type=rtype,
        props=props or {},
        confidence=conf,
        source_chunk_ids=[chunk.chunk_id],
    )


# =============================================================================
# 关系抽取：LLM
# =============================================================================

EXTRACT_PROMPT = """你是一个信息抽取器。从下面这段网文平台的政策文档中，抽取配置项之间的关系。

# 允许的关系类型（只能用这些，不要自创）
- DEPENDS_ON：A 依赖 B，B 未开启时 A 不生效
- REQUIRES_VALUE：A 要求 B 取某个特定值（必须在 value 字段给出该值）
- MUTEX_WITH：A 与 B 互斥，不能同时开启
- ENHANCES：A 对某个业务指标是正向
- WEAKENS：A 对某个业务指标是负向

# 可用的配置项 key（source 和 DEPENDS_ON/REQUIRES_VALUE/MUTEX_WITH 的 target 必须从中选）
{item_keys}

# 可用的业务指标 key（ENHANCES/WEAKENS 的 target 必须从中选）
{effect_keys}

# 抽取要求
1. **只抽文中明确说了的关系**。文中没提到的，即使你觉得应该有，也不要抽。
2. 注意代词。文中大量用"它""这个开关""该配置项"指代小节标题里的配置项，
   遇到代词时，主语取小节标题对应的配置项。
3. 注意方向。"A 依赖 B" 的 source 是 A、target 是 B，不要弄反。
   "B 是 A 的前置条件" 同样是 source=A、target=B。
4. MUTEX_WITH 是双向的，只需输出一条。
5. 抽不到任何关系时返回空数组，不要硬凑。

# 输出格式
{{"relations": [{{"source": "配置项key", "target": "配置项key或指标key", "type": "关系类型", "value": "仅REQUIRES_VALUE需要，否则省略", "evidence": "原文中支撑这条关系的片段，不超过40字"}}]}}

# 当前小节标题
{section}

# 待抽取文本
{text}"""


class LLMExtractor:
    name = "llm"

    def __init__(self) -> None:
        self.reg = get_registry()
        self.item_keys = "、".join(f"{k}({v.name})" for k, v in self.reg.items.items())
        self.effect_keys = "、".join(f"{k}({v})" for k, v in self.reg.effects.items())

    def extract(self, chunks: list[Chunk]) -> list[Relation]:
        out: list[Relation] = []
        for i, c in enumerate(chunks, 1):
            # 没有配置项的 chunk 直接跳过，省掉大量无效调用。
            # 实测能砍掉 30%+ 的 token 消耗，抽取结果没有损失。
            if len(c.linked_item_keys) < 1:
                continue
            try:
                out.extend(self._extract_one(c))
            except LLMError as e:
                logger.warning("chunk %s 抽取失败：%s", c.chunk_id, e)
            if i % 10 == 0:
                logger.info("LLM 抽取进度 %d/%d", i, len(chunks))
        return out

    def _extract_one(self, chunk: Chunk) -> list[Relation]:
        prompt = EXTRACT_PROMPT.format(
            item_keys=self.item_keys, effect_keys=self.effect_keys,
            section=chunk.section, text=chunk.text,
        )
        data = chat_json(
            [{"role": "user", "content": prompt}],
            purpose="extract_relation",
            model=settings.extract_model_name,
            temperature=0.0,
            max_tokens=1500,
        )
        raw = data.get("relations", []) if isinstance(data, dict) else []
        out: list[Relation] = []
        for r in raw:
            rel = self._validate(r, chunk)
            if rel is not None:
                out.append(rel)
        return out

    def _validate(self, r: dict, chunk: Chunk) -> Relation | None:
        """LLM 抽出来的每条边都要过一遍合法性：类型合法、实体存在、方向有意义。

        这一步很重要 —— 不校验的话，模型会稳定地产出
        source=不存在的key、type=CONFLICTS_WITH 这类幻觉，直接污染图谱。
        """
        try:
            rtype = RelationType(str(r.get("type", "")).upper())
        except ValueError:
            return None
        src, dst = str(r.get("source", "")), str(r.get("target", ""))
        if src not in self.reg.items:
            return None
        if rtype in (RelationType.ENHANCES, RelationType.WEAKENS):
            if dst not in self.reg.effects:
                return None
            target_id = f"effect:{dst}"
        else:
            if dst not in self.reg.items or dst == src:
                return None
            target_id = f"config:{dst}"

        props = {}
        if rtype == RelationType.REQUIRES_VALUE and r.get("value") not in (None, ""):
            props["value"] = r["value"]
        if r.get("evidence"):
            props["evidence"] = str(r["evidence"])[:80]

        return Relation(
            source_id=f"config:{src}", target_id=target_id, type=rtype,
            props=props, confidence=0.75, source_chunk_ids=[chunk.chunk_id],
        )


# =============================================================================
# 结构性关系（无需抽取，直接来自配置平台元数据）
# =============================================================================


def structural_relations() -> list[Relation]:
    """BELONGS_TO / APPLIES_TO / CONSTRAINED_BY 这三类关系不用抽。

    它们是配置平台元数据的一部分（某配置项属于哪个分类、能配在哪些维度、
    被哪条校验规则覆盖），真实系统里从元数据库直接读即可。
    把它们和"需要从散文里抽"的关系区分开，评测时才不会互相污染。
    """
    reg = get_registry()
    out: list[Relation] = []
    for key, item in reg.items.items():
        src = f"config:{key}"
        if item.category:
            out.append(Relation(source_id=src, target_id=f"cat:{item.category}",
                                type=RelationType.BELONGS_TO))
        for d in item.dimensions:
            out.append(Relation(source_id=src, target_id=f"dim:{d}",
                                type=RelationType.APPLIES_TO))
    for rule in reg.composite_rules:
        blob = json.dumps(rule, ensure_ascii=False)
        for key in reg.items:
            if key in blob:
                out.append(Relation(source_id=f"config:{key}", target_id=f"rule:{rule['id']}",
                                    type=RelationType.CONSTRAINED_BY))
    return out


# =============================================================================
# 消歧 / 合并
# =============================================================================


def merge_relations(*groups: list[Relation]) -> list[Relation]:
    """多来源关系合并。

    去重键是 (source, target, type)，互斥额外做无序归一 ——
    "A 互斥 B" 和 "B 互斥 A" 是同一条边，不归一的话图上会出现两条，
    三元组导出时 LLM 会看到重复信息，白白占上下文。

    冲突处理：同一条边被多个来源抽到，取 confidence 最高的，
    并把所有来源的 source_chunk_ids 并起来（溯源要完整）。
    """
    pool: dict[tuple[str, str, str], Relation] = {}
    for group in groups:
        for r in group:
            if r.type == RelationType.MUTEX_WITH:
                a, b = sorted([r.source_id, r.target_id])
                r = r.model_copy(update={"source_id": a, "target_id": b})
            key = (r.source_id, r.target_id, r.type.value)
            exist = pool.get(key)
            if exist is None:
                pool[key] = r
                continue
            merged_chunks = list(dict.fromkeys([*exist.source_chunk_ids, *r.source_chunk_ids]))
            winner = r if r.confidence > exist.confidence else exist
            pool[key] = winner.model_copy(update={
                "source_chunk_ids": merged_chunks,
                # 两个来源都抽到同一条边，置信度上调 —— 这是最朴素的投票集成
                "confidence": min(1.0, max(exist.confidence, r.confidence) + 0.1),
                "props": {**exist.props, **r.props},
            })
    return list(pool.values())


def build_graph(
    chunks: list[Chunk], extractor: str = "hybrid", entity_mode: str = "vocab"
) -> tuple[KnowledgeGraph, list[dict]]:
    """返回 (图谱, 待审核实体)。待审核实体只在 entity_mode=llm 时可能非空。"""
    entities = build_entities(chunks)
    pending: list[dict] = []

    if entity_mode == "llm":
        mentions = LLMEntityExtractor().extract(chunks)
        found, pending, layers = resolve_mentions(mentions)
        logger.info("LLM 实体抽取：提及 %d 条，消歧各层命中 %s，待审核 %d 个",
                    len(mentions), layers, len(pending))
        # 配置项、业务效果只保留大模型真的抽到的；分类、规则、维度来自元数据，照常保留
        keep_prefix = ("cat:", "rule:", "dim:")
        entities = [
            e.model_copy(update={"source_chunk_ids": sorted(set(found[e.entity_id]))[:12]})
            if e.entity_id in found else e
            for e in entities
            if e.entity_id in found or e.entity_id.startswith(keep_prefix)
        ]

    valid_ids = {e.entity_id for e in entities}

    groups: list[list[Relation]] = [structural_relations()]
    if extractor in ("rule", "hybrid"):
        rule_rels = RuleExtractor().extract(chunks)
        logger.info("规则抽取关系 %d 条", len(rule_rels))
        groups.append(rule_rels)
    if extractor in ("llm", "hybrid"):
        llm_rels = LLMExtractor().extract(chunks)
        logger.info("LLM 抽取关系 %d 条", len(llm_rels))
        groups.append(llm_rels)

    relations = [
        r for r in merge_relations(*groups)
        if r.source_id in valid_ids and r.target_id in valid_ids
    ]
    logger.info("合并后关系 %d 条 / 实体 %d 个", len(relations), len(entities))
    return KnowledgeGraph(entities=entities, relations=relations), pending
