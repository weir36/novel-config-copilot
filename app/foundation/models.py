"""全链路数据契约。

离线产物、在线检索、Agent 工具调用、API 出入参共用这一套模型，
这样任何一个环节的字段变更都会在其他环节立刻暴露出来。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# 离线产物
# =============================================================================


class Chunk(BaseModel):
    """语义切块。切块单位是「一个配置项的完整描述」或「一个业务语义单元」，不是固定窗口。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str = ""
    text: str
    # 该 chunk 提到的全部配置项 key，用作图谱->文本的桥接（宽口径）
    linked_item_keys: list[str] = Field(default_factory=list)
    # 该 chunk **主要在讲**的配置项 —— 出现在小节标题里的那些。
    # 宽口径会把"顺带提了一句"也算命中，评测时严重虚高；
    # 严格口径回答的是"有没有召回到真正定义/解释这个配置项的那一段"。
    primary_item_keys: list[str] = Field(default_factory=list)
    char_len: int = 0
    strategy: Literal["semantic", "sliding"] = "semantic"


class EntityType(str, Enum):
    CONFIG_ITEM = "ConfigItem"
    EFFECT = "Effect"
    DIMENSION = "Dimension"
    CATEGORY = "Category"
    RULE = "Rule"
    SCENARIO = "Scenario"


class RelationType(str, Enum):
    DEPENDS_ON = "DEPENDS_ON"          # A 依赖 B
    REQUIRES_VALUE = "REQUIRES_VALUE"  # A 要求 B 取特定值
    MUTEX_WITH = "MUTEX_WITH"          # A 与 B 互斥（双向）
    ENHANCES = "ENHANCES"              # A 增强某业务效果
    WEAKENS = "WEAKENS"                # A 削弱某业务效果
    APPLIES_TO = "APPLIES_TO"          # A 可作用于某维度
    BELONGS_TO = "BELONGS_TO"          # A 属于某分类
    CONSTRAINED_BY = "CONSTRAINED_BY"  # A 受某条复合规则约束
    USED_IN = "USED_IN"                # A 被某个典型方案使用


class Entity(BaseModel):
    entity_id: str                       # 消歧后的稳定 ID，如 config:full_attendance_bonus
    name: str                            # 标准名
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    # 供实体向量索引使用的拼接文本
    embed_text: str = ""
    # 溯源：这个实体是从哪些 chunk 抽出来的
    source_chunk_ids: list[str] = Field(default_factory=list)
    props: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    source_id: str
    target_id: str
    type: RelationType
    # 关系上的取值约束，如 REQUIRES_VALUE 的 {"value": "advanced"}
    props: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# =============================================================================
# 在线检索
# =============================================================================


class ScoredEntity(BaseModel):
    entity: Entity
    score: float
    # 命中方式：linked=query 直接链接到的种子实体，expanded=图谱扩展出来的
    hit_type: Literal["linked", "expanded"] = "linked"
    hop: int = 0


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    vector_rank: int = -1
    bm25_rank: int = -1
    source: Literal["vector", "bm25", "hybrid"] = "hybrid"


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    note: str = ""

    def to_text(self) -> str:
        base = f"({self.subject}) -[{self.predicate}]-> ({self.object})"
        return f"{base} {self.note}".strip()


class RetrievalResult(BaseModel):
    query: str
    seed_entities: list[ScoredEntity] = Field(default_factory=list)
    subgraph_entities: list[ScoredEntity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    chunks: list[ScoredChunk] = Field(default_factory=list)
    # 被上下文缓存剪掉的实体（多轮去重）
    pruned_entity_ids: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)

    def context_text(self) -> str:
        parts: list[str] = []
        if self.triples:
            parts.append(
                "【知识图谱关系】\n" + "\n".join(f"- {t.to_text()}" for t in self.triples)
            )
        if self.chunks:
            body = "\n\n".join(
                f"[{i + 1}] 来源：{c.chunk.doc_title} / {c.chunk.section}\n{c.chunk.text}"
                for i, c in enumerate(self.chunks)
            )
            parts.append("【领域知识片段】\n" + body)
        return "\n\n".join(parts)


# =============================================================================
# 意图
# =============================================================================


class Intent(str, Enum):
    EXPLAIN = "explain"        # 解释类：某配置项是什么、为什么没生效、为什么被拦
    RECOMMEND = "recommend"    # 推荐类：给我一套满足 X 的方案
    OPTIMIZE = "optimize"      # 优化类：在现有配置上调整
    OUT_OF_SCOPE = "out_of_scope"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float
    # 兜底路径：低置信时走的策略
    fallback_used: bool = False
    scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0


# =============================================================================
# 规则校验
# =============================================================================


class ViolationSeverity(str, Enum):
    ERROR = "error"
    WARN = "warn"


class Violation(BaseModel):
    rule_id: str
    severity: ViolationSeverity
    message: str
    involved_items: list[str] = Field(default_factory=list)
    suggestion: str = ""


class ValidationReport(BaseModel):
    passed: bool
    violations: list[Violation] = Field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == ViolationSeverity.ERROR]

    @property
    def warns(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == ViolationSeverity.WARN]

    def to_text(self) -> str:
        if not self.violations:
            return "校验通过：未发现依赖缺失、互斥冲突或复合规则违规。"
        lines = []
        for v in self.violations:
            tag = "❌ ERROR" if v.severity == ViolationSeverity.ERROR else "⚠️ WARN"
            line = f"{tag} [{v.rule_id}] {v.message}"
            if v.suggestion:
                line += f"　建议：{v.suggestion}"
            lines.append(line)
        return "\n".join(lines)


class ConfigPlan(BaseModel):
    """一套配置方案：配置项 key -> 取值，外加生效维度。"""

    scenario: str = ""
    values: dict[str, Any] = Field(default_factory=dict)
    dimensions: dict[str, str] = Field(default_factory=dict)  # item_key -> dimension key
    rationale: dict[str, str] = Field(default_factory=dict)   # item_key -> 为什么这么配


# =============================================================================
# 会话与 Agent
# =============================================================================


class Turn(BaseModel):
    turn_id: int
    question: str
    answer: str = ""
    intent: Intent = Intent.EXPLAIN
    entity_ids: list[str] = Field(default_factory=list)
    item_keys: list[str] = Field(default_factory=list)
    # 这一轮的"话题焦点"：问题里原样提到的实体；追问时会继承上一轮的焦点
    focus_entity_ids: list[str] = Field(default_factory=list)
    is_followup: bool = False
    summary: str = ""


class FollowupResult(BaseModel):
    """追问识别结果：这一句是不是接着上文问的，以及该带上哪几轮历史。"""
    is_followup: bool = False
    reasons: list[str] = Field(default_factory=list)
    related_turn_ids: list[int] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    # 会话级累积的实体集合，用于 GraphRAG 扩展时剪枝
    seen_entity_ids: list[str] = Field(default_factory=list)
    # 会话级工作区：用户正在编辑的配置方案
    working_plan: ConfigPlan = Field(default_factory=ConfigPlan)
    rolling_summary: str = ""


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    ok: bool
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class AgentStep(BaseModel):
    step: int
    thought: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


class AgentTrace(BaseModel):
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str = ""
    stopped_reason: Literal["final", "max_steps", "error"] = "final"
    total_latency_ms: float = 0.0


class ChatRequest(BaseModel):
    session_id: str
    message: str
    # 调试开关：强制指定意图/关闭 Agent，评测脚本会用
    force_intent: Intent | None = None
    use_agent: bool | None = None
    retrieval_mode: Literal["graphrag", "hybrid", "vector", "bm25"] = "graphrag"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: IntentResult
    retrieval: RetrievalResult | None = None
    agent_trace: AgentTrace | None = None
    validation: ValidationReport | None = None
    plan: ConfigPlan | None = None
    followup: FollowupResult | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
