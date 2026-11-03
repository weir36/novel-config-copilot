"""索引与 embedder 一致性校验：降级到 tfidf 后不能拿 bge 建的索引硬查。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.retrieval import embedder
from app.retrieval.embedder import IndexMismatchError, check_index_compat
from app.retrieval.vector_store import VectorStore

BGE = {"backend": "bge", "dim": 512, "model": "BAAI/bge-small-zh-v1.5"}
TFIDF = {"backend": "tfidf", "dim": 132, "model": "builtin-tfidf"}


def _write_meta(d, meta):
    (d / "embedder_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_compat_passes_when_same(tmp_path, monkeypatch):
    _write_meta(tmp_path, BGE)
    monkeypatch.setattr(embedder, "embedder_meta", lambda: dict(BGE))
    check_index_compat(tmp_path)


def test_compat_raises_on_fallback(tmp_path, monkeypatch):
    _write_meta(tmp_path, BGE)
    monkeypatch.setattr(embedder, "embedder_meta", lambda: dict(TFIDF))
    with pytest.raises(IndexMismatchError) as ei:
        check_index_compat(tmp_path)
    msg = str(ei.value)
    assert "backend" in msg and "dim" in msg and "app.indexing.build" in msg


def test_compat_raises_on_model_change_same_dim(tmp_path, monkeypatch):
    """维度相同但模型不同，向量空间一样不可比。"""
    _write_meta(tmp_path, BGE)
    monkeypatch.setattr(embedder, "embedder_meta", lambda: {**BGE, "model": "other/model"})
    with pytest.raises(IndexMismatchError, match="model"):
        check_index_compat(tmp_path)


def test_compat_skips_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(embedder, "embedder_meta", lambda: dict(TFIDF))
    check_index_compat(tmp_path)  # 没有索引也没有 meta：什么都不做


def _store(dim: int) -> VectorStore:
    s = VectorStore("t")
    s.ids, s.payloads = ["a", "b"], [{}, {}]
    s.vectors = np.eye(2, dim, dtype=np.float32)
    s._faiss_index = None
    return s


def test_vector_search_rejects_wrong_dim():
    s = _store(4)
    with pytest.raises(ValueError, match="维"):
        s.search(np.ones(3), top_k=1)
    with pytest.raises(ValueError, match="维"):
        s.search_multi(np.ones((2, 3)), top_k=1)
    assert s.search(np.array([1, 0, 0, 0]), top_k=1)[0][0] == "a"
