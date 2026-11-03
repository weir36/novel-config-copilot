"""Embedding 层。

两个后端：
  - BGEEmbedder：BAAI/bge-small-zh-v1.5，线上用。注意 bge 系列要求 **查询侧加指令前缀、
    文档侧不加**，混用会掉点，这里显式区分 encode_query / encode_docs。
  - TfidfEmbedder：纯 numpy 实现的 jieba 分词 + 字符 2-gram 混合 TF-IDF，再用 SVD 降维。
    没装 torch 的机器上自动启用，保证评测链路仍然能跑出有意义的语义召回对比。

两者都归一化到单位长度，这样内积就是余弦相似度，向量库统一用 IndexFlatIP 语义。
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import numpy as np

from app.foundation.settings import settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?|[一-鿿]")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BaseEmbedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def encode_docs(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray: ...

    def fit(self, corpus: list[str]) -> None:  # 只有 tfidf 需要
        return None

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> bool:
        return True


# =============================================================================
# BGE
# =============================================================================


class BGEEmbedder(BaseEmbedder):
    name = "bge"

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        logger.info("BGEEmbedder ready: %s dim=%d", self.model_name, self.dim)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = settings.bge_query_instruction + text
        vec = self.model.encode([prefixed], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0]


# =============================================================================
# TF-IDF fallback
# =============================================================================


class TfidfEmbedder(BaseEmbedder):
    """jieba 词 + 字符 2-gram 的混合 TF-IDF，SVD 降维后归一化。

    中文短文本上纯词级 TF-IDF 对未登录词很脆（"pk玩法" 切不开就废了），
    混入字符 2-gram 能把这部分补回来，代价是词表变大，语料只有几百 chunk 所以无所谓。
    """

    name = "tfidf"

    def __init__(self, dim: int | None = None) -> None:
        self.target_dim = dim or settings.embedding_dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (vocab, dim)
        self.dim = self.target_dim
        try:
            import jieba  # noqa: PLC0415

            jieba.setLogLevel(logging.WARNING)
            self._jieba = jieba
        except ImportError:
            self._jieba = None

    # ---------------- 分词 ----------------
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        toks: list[str] = []
        if self._jieba is not None:
            toks.extend(t.strip() for t in self._jieba.lcut(text) if t.strip())
        else:
            toks.extend(_TOKEN_RE.findall(text))
        # 字符 2-gram（只对中文片段）
        han = re.findall(r"[一-鿿]{2,}", text)
        for seg in han:
            toks.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        return [t for t in toks if t and not t.isspace()]

    # ---------------- 训练 ----------------
    def fit(self, corpus: list[str]) -> None:
        docs_tokens = [self._tokenize(t) for t in corpus]
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        # 去掉只出现一次的长尾，压一压词表
        self.vocab = {tok: i for i, (tok, c) in enumerate(df.most_common()) if c >= 1}
        n_docs = max(len(corpus), 1)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, i in self.vocab.items():
            idf[i] = math.log((1 + n_docs) / (1 + df[tok])) + 1.0
        self.idf = idf

        raw = self._raw_matrix(docs_tokens)
        # 截断 SVD：对 raw^T raw 做特征分解，语料小所以直接用 numpy
        k = min(self.target_dim, raw.shape[0], raw.shape[1])
        try:
            _u, _s, vt = np.linalg.svd(raw, full_matrices=False)
            self.components = np.ascontiguousarray(vt[:k].T).astype(np.float32)
        except np.linalg.LinAlgError:  # 极端情况下退化为不降维
            self.components = np.eye(raw.shape[1], dtype=np.float32)
        self.dim = int(self.components.shape[1])
        logger.info(
            "TfidfEmbedder fitted: vocab=%d docs=%d dim=%d", len(self.vocab), len(corpus), self.dim
        )

    def _raw_matrix(self, docs_tokens: list[list[str]]) -> np.ndarray:
        assert self.idf is not None
        mat = np.zeros((len(docs_tokens), len(self.vocab)), dtype=np.float32)
        for r, toks in enumerate(docs_tokens):
            tf = Counter(t for t in toks if t in self.vocab)
            if not tf:
                continue
            max_tf = max(tf.values())
            for tok, c in tf.items():
                mat[r, self.vocab[tok]] = (0.5 + 0.5 * c / max_tf) * self.idf[self.vocab[tok]]
        return l2_normalize(mat)

    # ---------------- 推理 ----------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.components is None or self.idf is None:
            raise RuntimeError("TfidfEmbedder 尚未 fit，请先跑离线建库")
        raw = self._raw_matrix([self._tokenize(t) for t in texts])
        return l2_normalize(raw @ self.components)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    # ---------------- 持久化 ----------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"vocab": self.vocab, "idf": self.idf, "components": self.components}, f
            )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("rb") as f:
            obj = pickle.load(f)
        self.vocab, self.idf, self.components = obj["vocab"], obj["idf"], obj["components"]
        self.dim = int(self.components.shape[1])
        return True


# =============================================================================
# 工厂
# =============================================================================

_EMBEDDER: BaseEmbedder | None = None
TFIDF_STATE_PATH = settings.index_dir / "tfidf_state.pkl"


def get_embedder(force: str | None = None) -> BaseEmbedder:
    """单例。auto 模式下优先 bge，import 失败则降级 tfidf 并自动加载已训练状态。"""
    global _EMBEDDER
    if _EMBEDDER is not None and force is None:
        return _EMBEDDER

    choice = force or settings.embedding_backend
    emb: BaseEmbedder | None = None
    if choice in ("auto", "bge"):
        try:
            emb = BGEEmbedder()
        except Exception as e:  # noqa: BLE001 - 降级是预期路径
            if choice == "bge":
                raise
            logger.warning("bge 不可用（%s），降级到内置 TF-IDF", type(e).__name__)
    if emb is None:
        emb = TfidfEmbedder()
        if emb.load(TFIDF_STATE_PATH):
            logger.info("TfidfEmbedder 已从 %s 恢复", TFIDF_STATE_PATH.name)

    if force is None:
        _EMBEDDER = emb
    return emb


def embedder_meta() -> dict:
    e = get_embedder()
    return {"backend": e.name, "dim": e.dim, "model": getattr(e, "model_name", "builtin-tfidf")}


def dump_meta(path: Path) -> None:
    path.write_text(json.dumps(embedder_meta(), ensure_ascii=False, indent=2), encoding="utf-8")


class IndexMismatchError(RuntimeError):
    """建库用的 embedder 和当前运行的 embedder 不一致。"""


def check_index_compat(index_dir: Path | None = None) -> None:
    """启动/热加载时比对 embedder_meta.json 与当前 embedder，不一致就立刻报错。

    get_embedder() 的降级是静默的：建库时用 bge（512 维），运行时 torch 没装上就会
    退到 tfidf（维度取决于语料条数）。不在这里拦住的话，要等第一条查询打到向量库
    才会以维度不匹配崩掉；就算维度碰巧相同，两个向量空间也不可比，召回结果是错的。
    """
    index_dir = index_dir or settings.index_dir
    meta_path = index_dir / "embedder_meta.json"
    if not meta_path.exists():
        if (index_dir / "chunks_vectors.npy").exists():
            logger.warning("索引缺少 %s，无法校验 embedder 一致性，建议重建索引", meta_path.name)
        return
    built = json.loads(meta_path.read_text(encoding="utf-8"))
    current = embedder_meta()
    diffs = [
        f"{k}: 建库={built.get(k)!r} 当前={current.get(k)!r}"
        for k in ("backend", "dim", "model")
        if built.get(k) != current.get(k)
    ]
    if diffs:
        raise IndexMismatchError(
            "索引与当前 embedder 不一致（" + "；".join(diffs) + "）。"
            "请在当前环境重建索引：python -m app.indexing.build，"
            "或修复环境让 embedding 后端与建库时一致（EMBEDDING_BACKEND / 安装 sentence-transformers）。"
        )
