"""意图分类器选型：只用训练集（留一交叉验证）+ 调参集，不读测试集。

    python -m eval.intent_model_select

对比向量部分的几种做法，关键词先验和越界规则保持不变：
  knn       现在的做法：top-k 近邻按相似度平方加权投票
  logreg    逻辑回归：学出"哪个方向区分意图"，不被话题相似度带偏
  centroid  每类取均值向量，按到类中心的相似度打分
"""
from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.foundation.models import Intent
from app.foundation.settings import settings
from app.intent.classifier import KEYWORD_PRIORS, OOS_MARKERS, VectorKNNClassifier, _has_domain_signal, get_classifier

logging.disable(logging.WARNING)
LABELS = [i for i in Intent]


def load_tune() -> list[tuple[str, Intent]]:
    rows = [json.loads(line) for line in (settings.gold_dir / "intent_tune.jsonl").read_text(encoding="utf-8").splitlines()]
    return [(r["query"], Intent(r["label"])) for r in rows if "query" in r]


def vote_knn(clf: VectorKNNClassifier, X, y, q, k=5):
    sims = X @ q
    out = np.zeros(len(LABELS))
    for i in np.argsort(-sims)[:k]:
        out[LABELS.index(y[i])] += max(float(sims[i]) - clf.floor, 0.0) ** 2
    return out / (out.sum() or 1.0), float(sims.max()) - clf.floor


def make_logreg(C):
    def vote(clf, X, y, q):
        model = LogisticRegression(C=C, max_iter=2000).fit(X, [LABELS.index(t) for t in y])
        proba = np.zeros(len(LABELS))
        proba[model.classes_] = model.predict_proba(q[None, :])[0]
        return proba, float((X @ q).max()) - clf.floor
    return vote


def vote_centroid(clf, X, y, q):
    cents = np.stack([X[[t == lab for t in y]].mean(0) for lab in LABELS])
    s = np.clip(cents @ q / np.linalg.norm(cents, axis=1) - clf.floor, 0, None) ** 2
    return s / (s.sum() or 1.0), float((X @ q).max()) - clf.floor


def _combine(clf, text, probs, knn_conf) -> Intent:
    """和 classify() 同一套先验 + 越界规则，只是向量投票部分由外面传进来。"""
    scores = {lab: float(p) for lab, p in zip(LABELS, probs)}
    q = text.strip().lower()
    for intent, words in KEYWORD_PRIORS.items():
        hits = sum(1 for w in words if w in q)
        if hits:
            scores[intent] += 0.22 * min(hits, 3)
    dom = _has_domain_signal(q)
    if any(mk in q for mk in OOS_MARKERS) and not dom:
        scores[Intent.OUT_OF_SCOPE] += 1.2
    if knn_conf < 0.18 and not dom:
        scores[Intent.OUT_OF_SCOPE] += 0.5
    if dom:
        scores[Intent.OUT_OF_SCOPE] *= 0.3
    total = sum(scores.values()) or 1.0
    best = max(scores.items(), key=lambda kv: kv[1])
    return Intent.EXPLAIN if best[1] / total < 0.30 and best[0] != Intent.OUT_OF_SCOPE else best[0]


def main() -> None:
    from app.retrieval.embedder import get_embedder
    clf = get_classifier()
    emb = get_embedder()
    X, y, texts = clf.vectors, clf.labels, clf.texts
    Q = np.stack([emb.encode_query(t) for t in texts])
    tune = load_tune()
    TQ = np.stack([emb.encode_query(t) for t, _ in tune])
    print(f"embedder={emb.name}  训练集 {len(texts)} 条（留一）  调参集 {len(tune)} 条\n")

    methods = {"knn": vote_knn, "centroid": vote_centroid}
    for C in (0.5, 1, 4, 16):
        methods[f"logreg C={C}"] = make_logreg(C)

    print(f"{'方法':<14}{'训练集留一':>10}{'调参集':>10}{'合计':>10}")
    for name, vote in methods.items():
        loo = 0
        for i in range(len(texts)):
            keep = np.arange(len(texts)) != i
            loo += _combine(clf, texts[i], *vote(clf, X[keep], [y[j] for j in range(len(y)) if keep[j]], Q[i])) == y[i]
        tn = sum(_combine(clf, t, *vote(clf, X, y, TQ[j])) == lab for j, (t, lab) in enumerate(tune))
        n = len(texts) + len(tune)
        print(f"{name:<14}{loo / len(texts):>10.3f}{tn / len(tune):>10.3f}{(loo + tn) / n:>10.3f}")


if __name__ == "__main__":
    main()
