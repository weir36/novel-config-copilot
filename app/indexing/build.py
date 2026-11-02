"""离线建库主流程。

    raw_docs/*.md
       ↓ 语义切块
    chunks.json ──┬─→ BGE embedding ─→ chunks 向量索引（FAISS / numpy）
                  ├─→ BM25 分词     ─→ chunks 关键词索引（ES / rank_bm25）
                  └─→ 实体（词表 或 大模型+消歧）+ 关系抽取 ─→ knowledge_graph.json
                                              ├─→ Neo4j / networkx
                                              └─→ entities 向量索引

跑一次大约几秒（规则抽取）到几分钟（LLM 抽取，取决于 chunk 数和模型速度）。
用法：
    python -m app.indexing.build                    # 默认 hybrid 抽取
    python -m app.indexing.build --extractor rule   # 纯规则，不调 LLM，完全离线
    python -m app.indexing.build --extractor llm    # 纯 LLM
    python -m app.indexing.build --entities llm     # 实体也用大模型抽 + 四层消歧
    python -m app.indexing.build --skip-graph       # 只重建文本索引
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.foundation.settings import settings
from app.retrieval.embedder import (
    TFIDF_STATE_PATH,
    TfidfEmbedder,
    dump_meta,
    get_embedder,
)
from app.retrieval.graph_store import KG_JSON_PATH, get_graph_store
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store import VectorStore
from app.foundation.models import Chunk, KnowledgeGraph
from app.indexing.chunker import chunk_all
from app.indexing.extract import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")
logger = logging.getLogger("indexing")

CHUNKS_PATH = settings.index_dir / "chunks.json"


def step_chunk() -> list[Chunk]:
    t0 = time.time()
    chunks = chunk_all()
    CHUNKS_PATH.write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lens = [c.char_len for c in chunks]
    logger.info(
        "[1/5] 切块完成：%d 块，平均 %.0f 字（min %d / max %d），耗时 %.1fs",
        len(chunks), sum(lens) / max(len(lens), 1), min(lens), max(lens), time.time() - t0,
    )
    no_item = [c.chunk_id for c in chunks if not c.linked_item_keys]
    if no_item:
        logger.info("      其中 %d 块未关联到任何配置项（多为流程/总述类内容）", len(no_item))
    return chunks


def step_fit_embedder(chunks: list[Chunk], kg: KnowledgeGraph | None) -> None:
    """TF-IDF 后端需要先在全量语料上 fit。BGE 不需要，这一步会自动跳过。"""
    emb = get_embedder()
    if not isinstance(emb, TfidfEmbedder):
        return
    corpus = [c.text for c in chunks]
    if kg:
        corpus.extend(e.embed_text for e in kg.entities if e.embed_text)
    # 意图种子也进语料，否则意图分类器的向量和检索向量不在同一空间
    intent_path = settings.gold_dir / "intent_train.json"
    if intent_path.exists():
        data = json.loads(intent_path.read_text(encoding="utf-8"))
        for label, samples in data.items():
            if not label.startswith("_"):
                corpus.extend(samples)
    emb.fit(corpus)
    emb.save(TFIDF_STATE_PATH)
    logger.info("      TF-IDF 已在 %d 条语料上拟合并保存", len(corpus))


def step_vector_index(chunks: list[Chunk]) -> None:
    t0 = time.time()
    emb = get_embedder()
    store = VectorStore("chunks")
    vecs = emb.encode_docs([c.text for c in chunks])
    store.build(
        ids=[c.chunk_id for c in chunks],
        vectors=vecs,
        payloads=[c.model_dump() for c in chunks],
    )
    store.save(settings.index_dir)
    logger.info("[3/5] 文本向量索引完成：%d 条，耗时 %.1fs", len(chunks), time.time() - t0)


def step_keyword_index(chunks: list[Chunk]) -> None:
    t0 = time.time()
    store = KeywordStore("chunks")
    store.build(
        ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        payloads=[c.model_dump() for c in chunks],
    )
    store.save(settings.index_dir)
    logger.info("[4/5] 关键词索引完成（%s）：%d 条，耗时 %.1fs",
                store.backend, len(chunks), time.time() - t0)


def step_graph(chunks: list[Chunk], extractor: str, entity_mode: str = "vocab") -> KnowledgeGraph:
    t0 = time.time()
    kg, pending = build_graph(chunks, extractor=extractor, entity_mode=entity_mode)
    KG_JSON_PATH.write_text(kg.model_dump_json(indent=2), encoding="utf-8")
    pending_path = settings.index_dir / "pending_entities.json"
    if pending:
        # 消歧对不上的实体不进图，留给人审：可能是词表漏了的新配置项，也可能是抽错了
        pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("      %d 个实体未能消歧，已写入 %s 待人工审核", len(pending), pending_path.name)
    elif pending_path.exists():
        pending_path.unlink()

    by_type: dict[str, int] = {}
    for r in kg.relations:
        by_type[r.type.value] = by_type.get(r.type.value, 0) + 1
    logger.info(
        "[2/5] 图谱构建完成（实体=%s，关系=%s）：%d 实体 / %d 关系，耗时 %.1fs",
        entity_mode, extractor, len(kg.entities), len(kg.relations), time.time() - t0,
    )
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        logger.info("      %-18s %d", t, n)
    return kg


def step_write_graph(kg: KnowledgeGraph) -> None:
    store = get_graph_store()
    store.write(kg)
    logger.info("      图谱已写入 %s", store.backend)


def step_entity_index(kg: KnowledgeGraph) -> None:
    t0 = time.time()
    emb = get_embedder()
    ents = [e for e in kg.entities if e.embed_text]
    store = VectorStore("entities")
    store.build(
        ids=[e.entity_id for e in ents],
        vectors=emb.encode_docs([e.embed_text for e in ents]),
        payloads=[
            {
                "name": e.name, "type": e.type.value, "aliases": e.aliases,
                "description": e.description, "embed_text": e.embed_text, "props": e.props,
            }
            for e in ents
        ],
    )
    store.save(settings.index_dir)
    logger.info("[5/5] 实体向量索引完成：%d 个实体，耗时 %.1fs", len(ents), time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["rule", "llm", "hybrid"], default="hybrid")
    ap.add_argument("--entities", choices=["vocab", "llm"], default="vocab",
                    help="实体抽取方式：vocab=词表（默认），llm=大模型抽取 + 四层消歧")
    ap.add_argument("--skip-graph", action="store_true", help="只重建文本索引")
    args = ap.parse_args()

    t_all = time.time()
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    chunks = step_chunk()

    kg: KnowledgeGraph | None = None
    if not args.skip_graph:
        # 图谱要在 embedding fit 之前建，因为实体的 embed_text 也是语料的一部分
        kg = step_graph(chunks, args.extractor, args.entities)
    elif KG_JSON_PATH.exists():
        kg = KnowledgeGraph.model_validate_json(KG_JSON_PATH.read_text(encoding="utf-8"))

    step_fit_embedder(chunks, kg)
    step_vector_index(chunks)
    step_keyword_index(chunks)

    if kg is not None:
        step_write_graph(kg)
        step_entity_index(kg)

    dump_meta(settings.index_dir / "embedder_meta.json")
    (settings.index_dir / "build_info.json").write_text(
        json.dumps(
            {
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "extractor": args.extractor if not args.skip_graph else "(skipped)",
                "entities_mode": args.entities,
                "chunks": len(chunks),
                "entities": len(kg.entities) if kg else 0,
                "relations": len(kg.relations) if kg else 0,
                "elapsed_s": round(time.time() - t_all, 1),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("全部完成，总耗时 %.1fs，索引目录：%s", time.time() - t_all, settings.index_dir)


if __name__ == "__main__":
    main()
