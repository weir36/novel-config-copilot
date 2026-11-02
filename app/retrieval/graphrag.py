"""GraphRAG 两阶段检索编排。

    第一阶段  query --实体链接--> 种子实体 --N跳关系扩展--> 实体子图
    第二阶段  实体子图 --每个实体各发一次查询--> 向量召回 + BM25 召回 --加权融合--> 参考上下文

为什么要有第一阶段：这个业务里用户的问题几乎从不自包含。
"保底作品怎么配福利" 的正确答案需要用到不同文档里的内容（签约文档讲保底分成和 R003、
福利文档讲全勤奖和完本奖、全勤奖又牵出日更字数要求），而这几段文本彼此的字面重合度很低。
朴素 RAG 拿 query 去召回，只能捞到第一篇；GraphRAG 先在图上把
「加更活动 → 全勤奖 → 日更字数要求」这样的结构路径走出来，
再拿路径上每个实体分别去召回，三篇文档才都能进上下文。

retrieval_mode 支持四档，是给消融实验用的：
  vector   - 只用 query 做向量召回（朴素 RAG）
  bm25     - 只用 query 做关键词召回
  hybrid   - query 双路召回 + 融合，不过图
  graphrag - 完整两阶段
"""
from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np

from app.foundation.settings import settings
from app.retrieval.embedder import get_embedder
from app.retrieval import fusion
from app.retrieval.entity_linker import get_entity_linker
from app.retrieval.graph_store import get_graph_store, relations_to_triples
from app.retrieval.keyword_store import get_keyword_store
from app.retrieval.vector_store import get_vector_store
from app.foundation.models import Entity, RetrievalResult, ScoredEntity

logger = logging.getLogger(__name__)

RetrievalMode = Literal["graphrag", "hybrid", "vector", "bm25"]


class GraphRAGRetriever:
    def __init__(self) -> None:
        self.chunk_vs = get_vector_store("chunks")
        self.chunk_kw = get_keyword_store("chunks")
        self.linker = get_entity_linker()
        self.graph = get_graph_store()

    def refresh(self) -> None:
        from app.retrieval import entity_linker, graph_store, vector_store  # noqa: PLC0415

        vector_store.reset_stores()
        graph_store.reset_graph()
        entity_linker.reset_linker()
        self.__init__()

    # =========================================================================
    # 主入口
    # =========================================================================
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "graphrag",
        exclude_entity_ids: set[str] | None = None,
        top_k: int | None = None,
        fusion_strategy: str | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        result = RetrievalResult(query=query)
        top_k = top_k or settings.final_top_k

        if mode == "graphrag":
            self._stage1(query, result, exclude_entity_ids or set())
            t1 = time.perf_counter()
            result.latency_ms["stage1_graph"] = (t1 - t0) * 1000
            self._stage2_from_subgraph(query, result, top_k, fusion_strategy)
            result.latency_ms["stage2_retrieval"] = (time.perf_counter() - t1) * 1000
        else:
            self._single_stage(query, result, mode, top_k, fusion_strategy)
            result.latency_ms["stage2_retrieval"] = (time.perf_counter() - t0) * 1000

        result.latency_ms["total"] = (time.perf_counter() - t0) * 1000
        return result

    # =========================================================================
    # 第一阶段：实体链接 + 图谱扩展
    # =========================================================================
    def _stage1(self, query: str, result: RetrievalResult, exclude: set[str]) -> None:
        # 注意 exclude 只用于「扩展剪枝」，不用于「种子链接」：
        # 用户这一轮又提到了某个实体，说明它仍是焦点，不该因为上一轮提过就被剪掉。
        # 被剪掉的应该是「上一轮已经扩展过、这一轮没被直接提到」的外围实体。
        seeds = self.linker.link(query)
        result.seed_entities = seeds
        if not seeds:
            logger.info("实体链接未命中任何实体，本轮退化为 hybrid 召回：%s", query[:40])
            return

        seed_ids = [s.entity.entity_id for s in seeds]
        expand_exclude = exclude - set(seed_ids)
        visited, relations = self.graph.expand(
            seed_ids,
            hops=settings.graph_hops,
            max_nodes=settings.graph_max_nodes,
            exclude_ids=expand_exclude,
        )
        result.pruned_entity_ids = sorted(expand_exclude & set(visited.keys()) | expand_exclude)[:20]

        seed_score = {s.entity.entity_id: s.score for s in seeds}
        subgraph: list[ScoredEntity] = []
        for eid, (ent, hop, weight) in visited.items():
            base = seed_score.get(eid, 1.0)
            subgraph.append(
                ScoredEntity(
                    entity=ent,
                    score=float(base * weight),
                    hit_type="linked" if hop == 0 else "expanded",
                    hop=hop,
                )
            )
        subgraph.sort(key=lambda e: -e.score)
        result.subgraph_entities = subgraph

        ent_map: dict[str, Entity] = {e.entity.entity_id: e.entity for e in subgraph}
        result.triples = relations_to_triples(relations, ent_map)

    # =========================================================================
    # 第二阶段：子图 -> 文本
    # =========================================================================
    def _stage2_from_subgraph(
        self, query: str, result: RetrievalResult, top_k: int, fusion_strategy: str | None
    ) -> None:
        if not result.subgraph_entities:
            self._single_stage(query, result, "hybrid", top_k, fusion_strategy)
            return

        # 用子图实体的描述文本作为查询，而不是实体名 ——
        # 实体名太短（"全勤奖" 三个字），拿去做向量召回信息量不足；
        # 描述文本自带业务语境，召回质量明显更好。
        #
        # 但参与召回的实体必须**严格收敛**。这个域只有 16 个配置项，2 跳几乎能走遍全图，
        # 把整个子图都拿去发查询，等于宣布"所有文档都相关"，召回会退化成随机。
        # 所以这里只取子图中权重最高的若干个，其余只保留在三元组里给 LLM 读。
        entities = result.subgraph_entities[: settings.graph_query_entities]
        queries = [query] + [
            (e.entity.embed_text or f"{e.entity.name} {e.entity.description}").strip()
            for e in entities
        ]
        # 原始 query 权重恒为 1.0 且显著高于扩展实体：用户问的是这个，
        # 子图只是补充证据，不能反客为主。扩展实体按子图权重再整体打折。
        weights = np.array(
            [1.0] + [settings.graph_entity_query_weight * max(e.score, 0.05) for e in entities],
            dtype=np.float32,
        )

        # --- 向量路：加权多查询 max 聚合 ---
        embedder = get_embedder()
        qvecs = np.vstack([embedder.encode_query(q) for q in queries])
        vector_hits = self.chunk_vs.search_multi(
            qvecs, settings.vector_top_k * 2, weights=weights
        )

        # --- 关键词路：query 本身 + 实体标准名/别名 ---
        kw_queries = [query] + [
            " ".join([e.entity.name, *e.entity.aliases[:2]]) for e in entities
        ]
        bm25_hits = self.chunk_kw.search_multi(
            kw_queries, settings.bm25_top_k * 2, weights=weights.tolist()
        )

        chunks = fusion.fuse(
            vector_hits, bm25_hits, strategy=fusion_strategy, top_k=top_k * 2
        )
        chunks = fusion.dedup_by_text(chunks)

        # 图谱侧的先验：把结构信息注回文本排序。分三档，档位差距要拉开 ——
        # 早先只有"命中子图实体就 +8%"这一档，结果 2 跳外的实体（比如
        # 查 A 时扩展出来的某个 2 跳外的实体）能靠自身向量相似度排到第一，
        # 而真正**定义** A 的那一段反而掉出 top-k。这是明显的排序错误：
        # 用户问 A，讲 A 的那一段必须在前面。
        seed_keys = {
            e.entity.props.get("key") for e in result.seed_entities
            if e.entity.props.get("key")
        }
        subgraph_keys = {
            e.entity.props.get("key") for e in entities if e.entity.props.get("key")
        }
        for c in chunks:
            primary = set(c.chunk.primary_item_keys)
            if primary & seed_keys:
                c.score *= 1.60   # 这一段就是在讲用户问的那个配置项
            elif primary & subgraph_keys:
                c.score *= 1.18   # 在讲子图里的关联配置项
            overlap = len(set(c.chunk.linked_item_keys) & subgraph_keys)
            if overlap:
                c.score *= 1.0 + min(overlap, 3) * 0.05  # 顺带提到，弱加成
        chunks.sort(key=lambda c: -c.score)
        result.chunks = chunks[:top_k]

    # =========================================================================
    # 单阶段（消融基线）
    # =========================================================================
    def _single_stage(
        self,
        query: str,
        result: RetrievalResult,
        mode: str,
        top_k: int,
        fusion_strategy: str | None,
    ) -> None:
        vector_hits: list = []
        bm25_hits: list = []
        if mode in ("vector", "hybrid"):
            qvec = get_embedder().encode_query(query)
            vector_hits = self.chunk_vs.search(qvec, settings.vector_top_k * 2)
        if mode in ("bm25", "hybrid"):
            bm25_hits = self.chunk_kw.search(query, settings.bm25_top_k * 2)

        if mode == "vector":
            w_v, w_b = 1.0, 0.0
        elif mode == "bm25":
            w_v, w_b = 0.0, 1.0
        else:
            w_v = w_b = None

        chunks = fusion.fuse(
            vector_hits, bm25_hits, strategy=fusion_strategy,
            w_vector=w_v, w_bm25=w_b, top_k=top_k * 2,
        )
        result.chunks = fusion.dedup_by_text(chunks)[:top_k]


_RETRIEVER: GraphRAGRetriever | None = None


def get_retriever() -> GraphRAGRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = GraphRAGRetriever()
    return _RETRIEVER
