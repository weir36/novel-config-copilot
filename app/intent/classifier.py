"""意图识别。

作用有两个，第二个才是重点：
  1. 选提示词模版 —— 解释类要的是准确复述，推荐类要的是结构化输出，两者的思维路径不同；
  2. **在进入完整 RAG + Agent 链路之前把无效问题挡掉**。
     一次完整推理是 1 次 embedding + 2 次检索 + 最多 6 轮 Agent 调用，
     让"今天天气怎么样"跑完这一整套，既慢又烧钱。out_of_scope 直接短路返回。

实现上用「句向量 + 逻辑回归 + 关键词先验」的轻量组合而不是微调 BERT：

  - 这个任务只有 4 类，60 条种子样本上训一个逻辑回归不到 0.1 秒，
    再上微调模型边际收益很小，却要背一个 400MB 的权重和一套训练流程；
  - 更重要的是**可增量**：编辑反馈某类问题分错了，加几条种子样本，启动时自动重训，
    不需要单独的训练流程。这在冷启动阶段比准确率本身更值钱。

为什么不用 kNN 投票（最初的做法）：bge 编码的主要是"在说哪个配置项"，而意图取决于
"怎么问"。"全勤奖是什么"和"把全勤奖调到 6000"在向量空间里很近，近邻会按话题投票。
逻辑回归能学出区分意图的那几个方向。选型见 eval/intent_model_select.py（只用训练集留一
和调参集）：bge 下 kNN 0.813 → 逻辑回归 0.987，TF-IDF 下 0.947 → 0.973。
没装 sklearn 时自动退回 kNN。

如果确实要上微调模型（如 MacBERT-zh-small），只需实现 BaseIntentClassifier 换掉即可，
接口和评测脚本都不用动。
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod

import numpy as np

from app.foundation.settings import settings
from app.retrieval.embedder import get_embedder, l2_normalize
from app.foundation.models import Intent, IntentResult

logger = logging.getLogger(__name__)

# 强信号关键词：命中即给该意图一个先验加成。
# 这些词在评测里是 kNN 最容易搞错的边界情况的解药，
# 比如「为什么...不生效」和「...是什么意思」在向量空间里离得很近。
KEYWORD_PRIORS: dict[Intent, list[str]] = {
    Intent.RECOMMEND: [
        "给一套", "推荐一套", "生成", "设计一套", "设计一个", "怎么配", "配置方案", "出一套",
        "从零", "该开哪些", "有什么建议", "帮我出", "冷启动", "怎么搭",
    ],
    Intent.OPTIMIZE: [
        "现有", "现在这套", "当前配置", "当前这套", "在当前", "基础上", "优化", "调整",
        "改成", "改到", "调到", "往后调", "降到", "压一压", "加强", "已经开了", "其他不动", "别动",
    ],
    Intent.EXPLAIN: [
        "是什么", "什么意思", "区别", "解释", "取值范围", "多少", "要求是什么", "有什么好处",
        "依赖什么", "依赖哪些", "能一起开吗", "能同时开吗", "能拿", "能开吗", "什么条件", "门槛",
        # "为什么没拿到""为什么被拦"是作者和编辑最常问的，归在解释类：
        # 回答要定位到具体配置项和规则编号，提示词里单独约束
        "为什么", "没生效", "没拿到", "被拦", "怎么回事", "什么原因", "代表什么",
    ],
}

# out_of_scope 的负向词表：出现这些且没有任何配置领域词时，判定越界。
# 只根据训练集和调参集归纳，没有看测试集。
OOS_MARKERS = [
    "天气", "诗", "电影", "翻译", "笑话", "股票", "吃什么", "好吃", "怎么做",
    "小说", "主角", "剧情", "开头", "写作水平", "请假", "感谢信", "python",
    "手机", "去哪", "账号", "乘",
]


_LABELS = list(Intent)


class BaseIntentClassifier(ABC):
    name = "base"

    @abstractmethod
    def classify(self, query: str) -> IntentResult: ...


class VectorKNNClassifier(BaseIntentClassifier):
    name = "vector-knn"

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.labels: list[Intent] = []
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None
        self.floor = 0.0  # 相似度底噪，见 _load_seeds
        self.model = None  # 逻辑回归；sklearn 不可用时为 None，退回 kNN
        self._load_seeds()

    def _load_seeds(self) -> None:
        path = settings.gold_dir / "intent_train.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for label, samples in data.items():
            if label.startswith("_"):
                continue
            for s in samples:
                self.labels.append(Intent(label))
                self.texts.append(s)
        try:
            self.vectors = l2_normalize(get_embedder().encode_docs(self.texts))
            # 底噪 = 任意两条种子之间相似度的中位数。bge 下约 0.38（随便两句中文都有点像），
            # TF-IDF 下接近 0。kNN 只看比底噪高出的部分，否则 bge 的邻居相似度全挤在
            # 0.5~0.8，平方加权拉不开差距。不用标签，换 embedding 模型自动适配
            sims = self.vectors @ self.vectors.T
            self.floor = float(np.median(sims[~np.eye(len(sims), dtype=bool)]))
            logger.info("IntentClassifier: %d 条种子样本已编码，相似度底噪 %.2f", len(self.texts), self.floor)
        except Exception as e:  # noqa: BLE001 - tfidf 未 fit 时会走到这
            logger.warning("意图种子编码失败（%s），本轮仅用关键词先验", type(e).__name__)
            self.vectors = None
            return
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

            # C=1 是 sklearn 默认值；选型时 C=0.5/1/4 结果持平，没有专门调
            self.model = LogisticRegression(C=1.0, max_iter=2000).fit(
                self.vectors, [_LABELS.index(lab) for lab in self.labels]
            )
            self.name = "vector-logreg"
        except ImportError:
            logger.warning("未安装 scikit-learn，意图识别退回 kNN 投票")
            self.model = None
            self.name = "vector-knn"

    def refresh(self) -> None:
        self.labels, self.texts, self.vectors = [], [], None
        self._load_seeds()

    # ---------------- 主入口 ----------------
    def classify(self, query: str) -> IntentResult:
        t0 = time.perf_counter()
        q = query.strip()
        scores: dict[Intent, float] = {i: 0.0 for i in Intent}

        # --- 1. 向量部分：逻辑回归概率（或 kNN 加权投票），都归一化成和为 1 ---
        knn_conf = 0.0
        if self.vectors is not None and len(self.texts) >= self.k:
            qvec = get_embedder().encode_query(q)
            sims = self.vectors @ qvec
            if self.model is not None:
                proba = self.model.predict_proba(qvec[None, :])[0]
                for cls, p in zip(self.model.classes_, proba):
                    scores[_LABELS[cls]] += float(p)
            else:
                for idx in np.argsort(-sims)[: self.k]:
                    # 相似度做 softmax-ish 加权，比等权投票更能反映"有多像"
                    scores[self.labels[idx]] += max(float(sims[idx]) - self.floor, 0.0) ** 2
                total = sum(scores.values()) or 1.0
                for k in scores:
                    scores[k] /= total
            # 最近邻比底噪高出多少：太低说明和所有种子都不像，越界规则要用
            knn_conf = float(sims.max()) - self.floor

        # --- 2. 关键词先验 ---
        q_low = q.lower()
        for intent, words in KEYWORD_PRIORS.items():
            hits = sum(1 for w in words if w in q_low)
            if hits:
                scores[intent] += 0.22 * min(hits, 3)

        # --- 3. 越界判定 ---
        # 只有「命中越界词」且「没有任何配置领域词」时才判 out_of_scope，
        # 避免把"付费章节对留存的影响"这类带通用词的正常问题误杀
        if any(m in q_low for m in OOS_MARKERS) and not _has_domain_signal(q_low):
            scores[Intent.OUT_OF_SCOPE] += 1.2
        if knn_conf < 0.18 and not _has_domain_signal(q_low):
            scores[Intent.OUT_OF_SCOPE] += 0.5
        # 反过来，问题里有配置领域词时，kNN 投给越界的票要打折：
        # "精品审核怎么过"会因为"怎么"两个字被"怎么提高写作水平"这类越界样本吸走，
        # 但它明明在问配置项。领域词是比字面相似更可靠的信号
        if _has_domain_signal(q_low):
            scores[Intent.OUT_OF_SCOPE] *= 0.3

        total = sum(scores.values()) or 1.0
        norm = {k: v / total for k, v in scores.items()}
        best = max(norm.items(), key=lambda kv: kv[1])

        # 置信度过低时兜底到 explain：解释类的提示词最保守，
        # 拿不准意图时给一段准确的领域知识，比强行推荐一套可能不对的方案安全
        fallback = best[1] < 0.30 and best[0] != Intent.OUT_OF_SCOPE
        intent = Intent.EXPLAIN if fallback else best[0]

        return IntentResult(
            intent=intent,
            confidence=round(float(best[1]), 4),
            fallback_used=fallback,
            scores={k.value: round(float(v), 4) for k, v in norm.items()},
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


_DOMAIN_RE = re.compile(
    r"配置|配|签约|分成|保底|买断|全勤|完本|加更|日更|推荐位|首页|精品|扶持|新书|"
    r"免费阅读|付费|章节|入v|广告分成|频道|男频|女频|规则|r\d{3}|b\d{3}|w\d{4}|"
    r"_bonus|_ratio|_mode|_chapter|_plan|门槛|依赖|互斥|生效|被拦|留存|曝光",
    re.IGNORECASE,
)


def _has_domain_signal(q: str) -> bool:
    return bool(_DOMAIN_RE.search(q))


_CLF: BaseIntentClassifier | None = None


def get_classifier() -> BaseIntentClassifier:
    global _CLF
    if _CLF is None:
        _CLF = VectorKNNClassifier()
    return _CLF


def reset_classifier() -> None:
    global _CLF
    _CLF = None
