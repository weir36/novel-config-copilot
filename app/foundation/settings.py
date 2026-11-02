"""全局配置。

设计原则：每一类外部依赖都有 `auto` 模式 —— 探测得到就用真实后端，探测不到就降级到
本地实现，并在启动日志里明确打印实际选中的后端。这样 `make demo` 在没有 docker、
没有 API key 的机器上也能跑通全链路，而在完整环境下又是真实的分布式组件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------- 路径 ----------------
    root_dir: Path = ROOT
    data_dir: Path = ROOT / "data"
    raw_docs_dir: Path = ROOT / "data" / "raw_docs"
    gold_dir: Path = ROOT / "data" / "gold"
    config_store_dir: Path = ROOT / "data" / "config_store"
    index_dir: Path = ROOT / "data" / "index"

    # ---------------- 后端选择 ----------------
    vector_backend: Literal["auto", "faiss", "numpy"] = "auto"
    keyword_backend: Literal["auto", "es", "bm25"] = "auto"
    graph_backend: Literal["auto", "neo4j", "networkx"] = "auto"
    memory_backend: Literal["auto", "redis", "inproc"] = "auto"
    embedding_backend: Literal["auto", "bge", "tfidf"] = "auto"

    # ---------------- LLM ----------------
    # 兼容任意 OpenAI-compatible 网关：豆包/方舟、DeepSeek、Qwen、OpenAI、vLLM 自建
    llm_provider: Literal["auto", "openai", "mock"] = "auto"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4

    # 抽取用的模型可以和对话模型不同（离线任务可以用更强/更慢的）
    extract_model: str = ""

    # ---------------- Embedding ----------------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # tfidf 降维后的维度；bge-small-zh 实际为 512
    # bge 系列官方建议：查询侧加指令前缀，文档侧不加
    bge_query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # ---------------- 外部组件连接串 ----------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "livecopilot"
    neo4j_database: str = "neo4j"

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_s: int = 3600

    es_url: str = "http://localhost:9200"
    es_index: str = "live_config_chunks"

    # ---------------- 检索超参 ----------------
    # 第一阶段：query -> 实体
    entity_link_top_k: int = 5
    # 向量分数的阈值必须跟着 embedding 模型走：bge 不相关文本也有 ~0.4 的底分，
    # TF-IDF 的正常信号才 ~0.13。两个值都由 eval/calibrate_thresholds.py 算出
    # （越界问题最高分的 P95，只用训练集 + 调参集），换模型后要重新校准。
    entity_link_min_score_bge: float = 0.48
    entity_link_min_score_tfidf: float = 0.08
    graph_hops: int = 2
    graph_max_nodes: int = 40
    # 子图里真正参与第二阶段召回的实体数。这个值远小于 graph_max_nodes：
    # 子图整体进上下文（三元组），但只有权重最高的几个拿去发检索查询，
    # 否则在小规模图上 2 跳能走遍全图，召回会被稀释成随机。
    graph_query_entities: int = 6
    # 扩展实体查询相对原始 query 的权重折扣
    graph_entity_query_weight: float = 0.55

    # 第二阶段：实体子图 -> 文本
    vector_top_k: int = 12
    bm25_top_k: int = 12
    fusion_strategy: Literal["weighted", "rrf"] = "weighted"
    fusion_w_vector: float = 0.6
    fusion_w_bm25: float = 0.4
    rrf_k: int = 60
    final_top_k: int = 6
    final_min_score: float = 0.22

    # ---------------- 切块 ----------------
    chunk_target_chars: int = 240
    chunk_max_chars: int = 420
    chunk_overlap_chars: int = 40

    # ---------------- 会话 ----------------
    max_history_turns: int = 8
    summarize_after_turns: int = 4

    # ---------------- Agent ----------------
    agent_max_steps: int = 6
    agent_enabled: bool = True

    # ---------------- 服务 ----------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def extract_model_name(self) -> str:
        return self.extract_model or self.llm_model


settings = Settings()
settings.index_dir.mkdir(parents=True, exist_ok=True)
