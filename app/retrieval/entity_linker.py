"""实体链接：把口语化的 query 映射到标准图谱实体。

这是 GraphRAG 第一阶段，也是整条链路最容易出问题的一环。编辑说的是
"那个千字给钱的"、"全勤那个"，文档里写的是"保底金额（guarantee_amount）"、"全勤奖（full_attendance_bonus）"。
纯向量召回在这种短 query 上抖动很大，所以这里做了三路融合：

  1. 向量召回：实体向量索引（标准名 + 别名 + 描述 + 业务标签拼成的 embed_text）
  2. 精确匹配：query 里直接出现标准名或别名 —— 这类命中置信度最高，直接加满分
  3. 符号匹配：query 里出现配置项 key / 规则号，编辑贴后台报错时很常见

只靠向量的话，2 和 3 这两类"用户已经说得很准"的情况反而可能被稀释掉，实测掉 10+ 个点。
"""
from __future__ import annotations

import logging
import re

from app.foundation.settings import settings
from app.retrieval.embedder import get_embedder
from app.retrieval.vector_store import get_vector_store
from app.foundation.models import Entity, EntityType, ScoredEntity

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\bR\d{3}\b", re.IGNORECASE)


class EntityLinker:
    def __init__(self) -> None:
        self.store = get_vector_store("entities")
        self._alias_index: dict[str, str] = {}
        self._build_alias_index()

    def _build_alias_index(self) -> None:
        """别名 -> entity_id 的倒排。长别名优先，避免 '全勤' 抢走 '全勤奖' 的命中。"""
        self._alias_index = {}
        for eid, payload in zip(self.store.ids, self.store.payloads, strict=False):
            surfaces = [payload.get("name", "")] + list(payload.get("aliases", []))
            key = payload.get("props", {}).get("key")
            if key:
                surfaces.append(key)
            for s in surfaces:
                s = (s or "").strip().lower()
                if len(s) >= 2:
                    self._alias_index.setdefault(s, eid)

    def refresh(self) -> None:
        self.store = get_vector_store("entities")
        self._build_alias_index()

    def mentions(self, query: str) -> list[str]:
        """只返回 query 里**原样出现**的实体（标准名 / 别名 / key / 规则号），不做向量召回。

        追问识别要用它判断"这一句提没提到具体配置项"。这里不能用 link()：
        link() 的向量召回总会返回 top-k，哪怕用户一个配置项都没提。
        """
        consumed = query.lower()
        found: list[str] = []
        for surface in sorted(self._alias_index, key=len, reverse=True):
            if surface in consumed:
                eid = self._alias_index[surface]
                if eid not in found:
                    found.append(eid)
                consumed = consumed.replace(surface, " ")
        return found

    # ---------------- 主入口 ----------------
    def link(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[ScoredEntity]:
        top_k = top_k or settings.entity_link_top_k
        if min_score is None:
            min_score = (
                settings.entity_link_min_score_bge
                if get_embedder().name == "bge"
                else settings.entity_link_min_score_tfidf
            )
        exclude = exclude_ids or set()
        q_lower = query.lower()

        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        # --- 1. 向量召回 ---
        if self.store.ids:
            qvec = get_embedder().encode_query(query)
            for eid, score, payload in self.store.search(qvec, top_k * 3):
                scores[eid] = max(scores.get(eid, 0.0), float(score))
                payloads[eid] = payload

        # --- 2. 别名/标准名精确匹配 ---
        # 按 surface 长度降序，长的先匹配并"吃掉"对应片段，防止短别名重复计分
        consumed = q_lower
        for surface in sorted(self._alias_index, key=len, reverse=True):
            if surface in consumed:
                eid = self._alias_index[surface]
                # 命中越长的 surface 说明用户说得越准，给的加权越高
                bonus = 0.55 + min(len(surface), 8) / 8 * 0.35
                scores[eid] = max(scores.get(eid, 0.0), bonus)
                payloads.setdefault(eid, self.store.get(eid) or {})
                consumed = consumed.replace(surface, " ")

        # --- 3. 符号（配置项 key / 规则号）匹配 ---
        for sym in _KEY_RE.findall(query):
            eid = self._alias_index.get(sym.lower())
            if eid:
                scores[eid] = 1.0  # 用户直接贴了 key，没有歧义
                payloads.setdefault(eid, self.store.get(eid) or {})

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: list[ScoredEntity] = []
        for eid, score in ranked:
            if score < min_score or eid in exclude:
                continue
            payload = payloads.get(eid) or self.store.get(eid)
            if payload is None:
                continue
            out.append(
                ScoredEntity(entity=_payload_to_entity(eid, payload), score=score, hit_type="linked")
            )
            if len(out) >= top_k:
                break
        return out


def _payload_to_entity(entity_id: str, payload: dict) -> Entity:
    return Entity(
        entity_id=entity_id,
        name=payload.get("name", entity_id),
        type=EntityType(payload.get("type", EntityType.CONFIG_ITEM.value)),
        aliases=list(payload.get("aliases", [])),
        description=payload.get("description", ""),
        embed_text=payload.get("embed_text", ""),
        props=payload.get("props", {}),
    )


_LINKER: EntityLinker | None = None


def get_entity_linker() -> EntityLinker:
    global _LINKER
    if _LINKER is None:
        _LINKER = EntityLinker()
    return _LINKER


def reset_linker() -> None:
    global _LINKER
    _LINKER = None
