"""向量召回与关键词召回的融合排序。

两种策略：

  - weighted：先对两路分数各自做 min-max 归一化再加权求和。
    必须归一化的原因是 BM25 是无界的（可以是 0.3 也可以是 28），
    余弦相似度在 [-1,1]，不归一化直接加权等于让 BM25 单方面决定排序。
  - rrf（Reciprocal Rank Fusion）：只用排名不用分数，score = Σ 1/(k + rank)。
    对分数分布不敏感，是两路分数量纲差异大时更稳的选择。

默认用 weighted，因为在本项目的评测集上它比 RRF 高 2~3 个点 ——
原因是这个场景下 BM25 的高分命中（用户直接贴了配置项 key）确实应该压过向量，
而 RRF 会把这种"排名第一但分数断层领先"的信号抹平。具体数字见 eval/report。
"""
from __future__ import annotations

from app.foundation.settings import settings
from app.foundation.models import Chunk, ScoredChunk

Hit = tuple[str, float, dict]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def fuse(
    vector_hits: list[Hit],
    bm25_hits: list[Hit],
    strategy: str | None = None,
    w_vector: float | None = None,
    w_bm25: float | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[ScoredChunk]:
    strategy = strategy or settings.fusion_strategy
    w_v = settings.fusion_w_vector if w_vector is None else w_vector
    w_b = settings.fusion_w_bm25 if w_bm25 is None else w_bm25
    top_k = top_k or settings.final_top_k
    min_score = settings.final_min_score if min_score is None else min_score

    v_norm = _minmax([h[1] for h in vector_hits])
    b_norm = _minmax([h[1] for h in bm25_hits])

    pool: dict[str, dict] = {}

    def _slot(cid: str, payload: dict) -> dict:
        return pool.setdefault(
            cid,
            {"payload": payload, "v": 0.0, "b": 0.0, "vr": -1, "br": -1, "vraw": 0.0, "braw": 0.0},
        )

    for rank, ((cid, raw, payload), norm) in enumerate(zip(vector_hits, v_norm, strict=True)):
        s = _slot(cid, payload)
        s["v"], s["vr"], s["vraw"] = norm, rank, raw
    for rank, ((cid, raw, payload), norm) in enumerate(zip(bm25_hits, b_norm, strict=True)):
        s = _slot(cid, payload)
        s["b"], s["br"], s["braw"] = norm, rank, raw

    results: list[ScoredChunk] = []
    for cid, s in pool.items():
        if strategy == "rrf":
            k = settings.rrf_k
            score = 0.0
            if s["vr"] >= 0:
                score += w_v / (k + s["vr"] + 1)
            if s["br"] >= 0:
                score += w_b / (k + s["br"] + 1)
            score *= k  # 缩放回 0~1 附近，便于和阈值比较
        else:
            score = w_v * s["v"] + w_b * s["b"]

        source = "hybrid" if s["vr"] >= 0 and s["br"] >= 0 else ("vector" if s["vr"] >= 0 else "bm25")
        results.append(
            ScoredChunk(
                chunk=_payload_to_chunk(cid, s["payload"]),
                score=float(score),
                vector_score=float(s["vraw"]),
                bm25_score=float(s["braw"]),
                vector_rank=s["vr"],
                bm25_rank=s["br"],
                source=source,
            )
        )

    results.sort(key=lambda c: -c.score)
    # 阈值过滤，但至少保底 1 条 —— 全被滤掉时 LLM 就只能靠幻觉答了，
    # 宁可给一条低分文本让它自己判断"资料不足"
    filtered = [r for r in results if r.score >= min_score][:top_k]
    return filtered or results[:1]


def _payload_to_chunk(chunk_id: str, payload: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=payload.get("doc_id", ""),
        doc_title=payload.get("doc_title", ""),
        section=payload.get("section", ""),
        text=payload.get("text", ""),
        linked_item_keys=list(payload.get("linked_item_keys", [])),
        primary_item_keys=list(payload.get("primary_item_keys", [])),
        char_len=payload.get("char_len", 0),
        strategy=payload.get("strategy", "semantic"),
    )


def dedup_by_text(chunks: list[ScoredChunk], threshold: float = 0.88) -> list[ScoredChunk]:
    """近重复去除：同一配置项在多篇文档里被重复描述是常态，
    Jaccard 粗筛足够，不值得为此再跑一遍 embedding。"""
    kept: list[ScoredChunk] = []
    seen: list[set[str]] = []
    for c in chunks:
        grams = {c.chunk.text[i : i + 4] for i in range(max(len(c.chunk.text) - 3, 1))}
        if any(
            len(grams & s) / max(len(grams | s), 1) >= threshold for s in seen
        ):
            continue
        kept.append(c)
        seen.append(grams)
    return kept
