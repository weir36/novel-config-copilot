"""关键词索引（BM25）。

在这个业务里关键词召回不是"顺手加的"，是刚需：配置项 key（`full_attendance_bonus`）、
规则编号（`R001`）、取值枚举（`uhd_6000k`、`T1`）这类符号性 token，
语义向量几乎抓不住，但编辑恰恰经常直接把它们贴进 query 里。

后端优先 Elasticsearch（生产形态，支持分布式与增量更新），
探测不到则用 rank_bm25 的进程内实现，检索语义一致。
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

from app.foundation.settings import settings

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|R\d{3}|\d+(?:\.\d+)?%?")


def tokenize_zh(text: str) -> list[str]:
    """中英混排分词：jieba 切中文，正则额外保留下划线标识符与规则号。

    关键在于 `full_attendance_bonus` 这种 token 必须整体保留 —— jieba 会把它切成
    `paid`/`_`/`pk`/`_`/`enable`，切完就匹配不上了。所以符号 token 单独抽一遍再并进去。
    """
    text_l = text.lower()
    toks: list[str] = []
    try:
        import jieba  # noqa: PLC0415

        jieba.setLogLevel(logging.WARNING)
        toks.extend(t.strip() for t in jieba.lcut(text_l) if t.strip())
    except ImportError:
        toks.extend(re.findall(r"[一-鿿]|[a-z0-9_]+", text_l))
    toks.extend(m.lower() for m in _SYMBOL_RE.findall(text))
    return [t for t in toks if t and not t.isspace()]


class KeywordStore:
    def __init__(self, name: str = "chunks") -> None:
        self.name = name
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self._bm25 = None
        self._es = None
        self.backend = "bm25"
        if settings.keyword_backend in ("auto", "es"):
            self._es = self._try_es()
            if self._es is not None:
                self.backend = "es"
            elif settings.keyword_backend == "es":
                raise RuntimeError("keyword_backend=es 但 Elasticsearch 不可用")

    @staticmethod
    def _try_es():
        try:
            from elasticsearch import Elasticsearch  # noqa: PLC0415

            es = Elasticsearch(settings.es_url, request_timeout=3)
            if es.ping():
                return es
        except Exception:  # noqa: BLE001 - 探测失败即降级
            pass
        return None

    # ---------------- 构建 ----------------
    def build(self, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        self.ids, self.payloads = list(ids), list(payloads)
        if self.backend == "es":
            self._build_es(texts)
        else:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415

            self._bm25 = BM25Okapi([tokenize_zh(t) for t in texts])
        logger.info("KeywordStore[%s] built: n=%d backend=%s", self.name, len(ids), self.backend)

    def _build_es(self, texts: list[str]) -> None:
        from elasticsearch import helpers  # noqa: PLC0415

        idx = f"{settings.es_index}_{self.name}"
        if self._es.indices.exists(index=idx):
            self._es.indices.delete(index=idx)
        self._es.indices.create(
            index=idx,
            body={
                "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        # 预分词字段：把 tokenize_zh 的结果也存一份，
                        # 让 ES 和本地 BM25 用同一套切词，保证两个后端召回可比
                        "tokens": {"type": "text"},
                        "payload": {"type": "object", "enabled": False},
                    }
                },
            },
        )
        actions = [
            {
                "_index": idx,
                "_id": doc_id,
                "_source": {
                    "text": text,
                    "tokens": " ".join(tokenize_zh(text)),
                    "payload": payload,
                },
            }
            for doc_id, text, payload in zip(self.ids, texts, self.payloads, strict=True)
        ]
        helpers.bulk(self._es, actions)
        self._es.indices.refresh(index=idx)

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        if not self.ids:
            return []
        if self.backend == "es":
            return self._search_es(query, top_k)
        import numpy as np  # noqa: PLC0415

        scores = np.asarray(self._bm25.get_scores(tokenize_zh(query)), dtype=float)
        k = min(top_k, len(self.ids))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i]), self.payloads[i]) for i in top if scores[i] > 0]

    def _search_es(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        idx = f"{settings.es_index}_{self.name}"
        res = self._es.search(
            index=idx,
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": " ".join(tokenize_zh(query)),
                        "fields": ["tokens^2", "text"],
                    }
                },
            },
        )
        return [
            (h["_id"], float(h["_score"]), h["_source"]["payload"])
            for h in res["hits"]["hits"]
        ]

    def search_multi(
        self, queries: list[str], top_k: int, weights: list[float] | None = None
    ) -> list[tuple[str, float, dict]]:
        """多查询加权取 max 聚合，与 VectorStore.search_multi 对齐。

        BM25 这一路同样需要加权：实体名查询的原始 BM25 分数可能远高于用户 query
        （实体名是文档里的高频精确串），不打折就会盖过用户真正问的东西。
        """
        pool: dict[str, tuple[float, dict]] = {}
        for i, q in enumerate(queries):
            w = weights[i] if weights and i < len(weights) else 1.0
            for doc_id, raw, payload in self.search(q, top_k):
                score = raw * w
                if doc_id not in pool or score > pool[doc_id][0]:
                    pool[doc_id] = (score, payload)
        ranked = sorted(pool.items(), key=lambda kv: -kv[1][0])[:top_k]
        return [(doc_id, s, p) for doc_id, (s, p) in ranked]

    # ---------------- 持久化 ----------------
    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        (dirpath / f"kw_{self.name}_meta.json").write_text(
            json.dumps({"ids": self.ids, "payloads": self.payloads, "backend": self.backend},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        if self.backend == "bm25" and self._bm25 is not None:
            with (dirpath / f"kw_{self.name}_bm25.pkl").open("wb") as f:
                pickle.dump(self._bm25, f)

    def load(self, dirpath: Path) -> bool:
        mpath = dirpath / f"kw_{self.name}_meta.json"
        if not mpath.exists():
            return False
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        self.ids, self.payloads = meta["ids"], meta["payloads"]
        if self.backend == "bm25":
            bpath = dirpath / f"kw_{self.name}_bm25.pkl"
            if not bpath.exists():
                return False
            with bpath.open("rb") as f:
                self._bm25 = pickle.load(f)
        return True


_KW_STORES: dict[str, KeywordStore] = {}


def get_keyword_store(name: str = "chunks", auto_load: bool = True) -> KeywordStore:
    if name not in _KW_STORES:
        store = KeywordStore(name)
        if auto_load:
            store.load(settings.index_dir)
        _KW_STORES[name] = store
    return _KW_STORES[name]
