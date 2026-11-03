"""向量索引。

两套索引复用同一实现，但物理隔离成两张表：
  - chunks：文本段向量，第二阶段语义召回用
  - entities：实体向量（标准名 + 别名 + 描述 + 业务标签），第一阶段实体链接用

向量统一归一化，因此内积 == 余弦。FAISS 装了就用 IndexFlatIP，没装用 numpy 矩阵乘法
—— 语料量级在万级以内时两者延迟差异在微秒级，不值得为此引入编译依赖。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.foundation.settings import settings
from app.retrieval.embedder import l2_normalize

logger = logging.getLogger(__name__)


def _faiss_available() -> bool:
    if settings.vector_backend == "numpy":
        return False
    try:
        import faiss  # noqa: F401, PLC0415

        return True
    except ImportError:
        if settings.vector_backend == "faiss":
            raise
        return False


class VectorStore:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.vectors: np.ndarray | None = None
        self._faiss_index = None
        self.backend = "faiss" if _faiss_available() else "numpy"

    # ---------------- 构建 ----------------
    def build(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        assert len(ids) == len(payloads) == vectors.shape[0]
        self.ids = list(ids)
        self.payloads = list(payloads)
        self.vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._build_faiss()
        logger.info(
            "VectorStore[%s] built: n=%d dim=%d backend=%s",
            self.name, len(ids), self.vectors.shape[1], self.backend,
        )

    def _build_faiss(self) -> None:
        if self.backend != "faiss" or self.vectors is None or len(self.ids) == 0:
            return
        import faiss  # noqa: PLC0415

        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        self._faiss_index = index

    # ---------------- 检索 ----------------
    def _check_dim(self, q: np.ndarray) -> None:
        if q.shape[-1] != self.vectors.shape[1]:
            raise ValueError(
                f"VectorStore[{self.name}] 查询向量 {q.shape[-1]} 维，索引 {self.vectors.shape[1]} 维："
                "embedding 后端与建库时不一致，请重建索引"
            )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float, dict]]:
        if self.vectors is None or not self.ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        self._check_dim(q)
        q = l2_normalize(q)
        k = min(top_k, len(self.ids))

        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, k)
            pairs = list(zip(idxs[0].tolist(), scores[0].tolist(), strict=True))
        else:
            sims = (self.vectors @ q[0]).astype(np.float32)
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            pairs = [(int(i), float(sims[i])) for i in top]

        return [
            (self.ids[i], float(s), self.payloads[i]) for i, s in pairs if i >= 0
        ]

    def search_multi(
        self,
        query_vecs: np.ndarray,
        top_k: int,
        weights: np.ndarray | None = None,
        aggregate: str = "max",
    ) -> list[tuple[str, float, dict]]:
        """多查询向量聚合检索 —— 第二阶段用实体子图里的每个实体各发一次查询。

        取 max 而不是 mean 的原因：子图里的实体彼此相关但不同质，mean 会把
        「只和其中一个实体强相关」的关键文本段压下去，而那往往正是要找的东西。

        weights 必须在**归一化之后**、聚合之前乘到相似度上。
        早先的写法是把权重预乘进查询向量再传进来，但这里的 l2_normalize
        会把模长整个除掉，权重完全失效 —— 结果是用户的原始 query 和几十个
        扩展实体等权，query 信号被子图淹没，召回质量反而低于朴素 RAG。
        """
        if self.vectors is None or not self.ids or query_vecs.size == 0:
            return []
        qs = np.asarray(query_vecs, dtype=np.float32)
        self._check_dim(qs)
        qs = l2_normalize(qs)
        sims = self.vectors @ qs.T  # (n_docs, n_queries)
        if weights is not None:
            sims = sims * np.asarray(weights, dtype=np.float32)[None, :]
        agg = sims.max(axis=1) if aggregate == "max" else sims.mean(axis=1)
        k = min(top_k, len(self.ids))
        top = np.argpartition(-agg, k - 1)[:k]
        top = top[np.argsort(-agg[top])]
        return [(self.ids[i], float(agg[i]), self.payloads[i]) for i in top]

    def get(self, doc_id: str) -> dict | None:
        try:
            return self.payloads[self.ids.index(doc_id)]
        except ValueError:
            return None

    # ---------------- 持久化 ----------------
    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        np.save(dirpath / f"{self.name}_vectors.npy", self.vectors)
        (dirpath / f"{self.name}_meta.json").write_text(
            json.dumps({"ids": self.ids, "payloads": self.payloads}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, dirpath: Path) -> bool:
        vpath = dirpath / f"{self.name}_vectors.npy"
        mpath = dirpath / f"{self.name}_meta.json"
        if not vpath.exists() or not mpath.exists():
            return False
        self.vectors = np.load(vpath)
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        self.ids, self.payloads = meta["ids"], meta["payloads"]
        self._build_faiss()
        return True


_STORES: dict[str, VectorStore] = {}


def get_vector_store(name: str, auto_load: bool = True) -> VectorStore:
    if name not in _STORES:
        store = VectorStore(name)
        if auto_load:
            store.load(settings.index_dir)
        _STORES[name] = store
    return _STORES[name]


def reset_stores() -> None:
    _STORES.clear()
