"""实体链接阈值校准：用越界问题的分数当底噪，不看任何检索标签。

    python -m eval.calibrate_thresholds

思路：越界问题（"怎么提高写作水平"）本来就不该链接到任何配置项，所以它们和
"最像的那个实体"之间的向量分数，就是这个 embedding 模型的底噪。阈值定在底噪的
P95 上方，意味着 95% 的纯噪声会被挡在外面。

只用训练集 + 调参集的问题，测试集（intent_eval / retrieval_eval）一条都不读。
bge 和 TF-IDF 的分数分布完全不同，所以各算各的阈值。
"""
from __future__ import annotations

import json
import logging

import numpy as np

from app.foundation.models import KnowledgeGraph
from app.foundation.settings import settings
from app.retrieval.embedder import BGEEmbedder, TfidfEmbedder

logging.disable(logging.WARNING)


def load_queries() -> tuple[list[str], list[str]]:
    """返回 (越界问题, 领域内问题)，只来自训练集和调参集。"""
    noise, signal = [], []
    train = json.loads((settings.gold_dir / "intent_train.json").read_text(encoding="utf-8"))
    for label, samples in train.items():
        if label.startswith("_"):
            continue
        (noise if label == "out_of_scope" else signal).extend(samples)
    for line in (settings.gold_dir / "intent_tune.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "query" in row:
            (noise if row["label"] == "out_of_scope" else signal).append(row["query"])
    return noise, signal


def make_embedder(name: str, corpus: list[str]):
    if name == "bge":
        return BGEEmbedder()
    emb = TfidfEmbedder()
    emb.fit(corpus)  # 和 build.step_fit_embedder 用同一份语料
    return emb


def main() -> None:
    kg = KnowledgeGraph(**json.loads((settings.index_dir / "knowledge_graph.json").read_text(encoding="utf-8")))
    ents = [e for e in kg.entities if e.embed_text]
    chunks = json.loads((settings.index_dir / "chunks.json").read_text(encoding="utf-8"))
    noise_q, signal_q = load_queries()
    train = json.loads((settings.gold_dir / "intent_train.json").read_text(encoding="utf-8"))
    corpus = [c["text"] for c in chunks] + [e.embed_text for e in ents]
    corpus += [s for k, v in train.items() if not k.startswith("_") for s in v]

    print(f"越界问题 {len(noise_q)} 条（底噪），领域内问题 {len(signal_q)} 条，实体 {len(ents)} 个\n")
    result = {}
    for name in ("bge", "tfidf"):
        emb = make_embedder(name, corpus)
        ent_vecs = emb.encode_docs([e.embed_text for e in ents])
        top1 = lambda qs: np.array([float((ent_vecs @ emb.encode_query(q)).max()) for q in qs])  # noqa: E731
        noise, signal = top1(noise_q), top1(signal_q)
        threshold = round(float(np.percentile(noise, 95)), 2)
        result[name] = threshold
        print(f"[{name}]")
        print(f"  底噪（越界问题的最高分）  中位数 {np.median(noise):.2f}  P95 {np.percentile(noise, 95):.2f}  最大 {noise.max():.2f}")
        print(f"  信号（领域内问题的最高分）中位数 {np.median(signal):.2f}  P5  {np.percentile(signal, 5):.2f}  最小 {signal.min():.2f}")
        print(f"  → 阈值 = 底噪 P95 = {threshold}；领域内问题高于阈值的比例 {(signal >= threshold).mean():.0%}\n")
    print("写进 .env：")
    print(f"ENTITY_LINK_MIN_SCORE_BGE={result['bge']}")
    print(f"ENTITY_LINK_MIN_SCORE_TFIDF={result['tfidf']}")


if __name__ == "__main__":
    main()
