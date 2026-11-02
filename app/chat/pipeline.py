"""在线推理主链路编排。

    query
      ↓ 意图识别（out_of_scope 在这里短路，不进入后续任何一步）
      ↓ 追问识别：是追问才带相关的历史轮次、做实体剪枝；新话题两样都不做
      ↓ GraphRAG 两阶段检索
      ↓ 分流：解释类走单次生成，推荐/优化类走 Agent 循环
      ↓ 方案抽取 + 规则引擎兜底校验
      ↓ 回写会话状态
    answer

分流的依据是「这个问题需不需要依赖前序结果才能决定下一步查什么」。
解释类不需要 —— 一次检索就能拿全资料，再上 Agent 只是徒增 3~5 倍延迟和成本。
推荐/优化需要，所以给它们 Agent。
这个分流是整个系统里性价比最高的一个设计：它让大部分流量走廉价路径。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from app.agent.loop import get_agent
from app.agent.tools import detect_scope, validate_with_base
from app.validation.plan_extract import extract_plan
from app.foundation.settings import settings
from app.intent.classifier import get_classifier
from app.generation.llm_client import get_llm
from app.generation.prompts import OUT_OF_SCOPE_REPLY, build_prompt
from app.session.session import get_session_manager
from app.retrieval.graph_store import get_graph_store
from app.retrieval.graphrag import get_retriever
from app.foundation.models import (
    FollowupResult,
    ChatRequest,
    ChatResponse,
    ConfigPlan,
    Intent,
    IntentResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# 需要 Agent 多步推理的意图
AGENT_INTENTS = {Intent.RECOMMEND, Intent.OPTIMIZE}
# 会产出配置方案、必须过规则校验的意图
PLAN_INTENTS = {Intent.RECOMMEND, Intent.OPTIMIZE}


def handle_chat(req: ChatRequest) -> ChatResponse:
    t_start = time.perf_counter()
    latency: dict[str, float] = {}
    sessions = get_session_manager()
    state = sessions.load(req.session_id)

    # ---------------- 1. 意图识别 ----------------
    t0 = time.perf_counter()
    if req.force_intent is not None:
        intent_res = IntentResult(intent=req.force_intent, confidence=1.0)
    else:
        intent_res = get_classifier().classify(req.message)
    latency["intent"] = (time.perf_counter() - t0) * 1000

    if intent_res.intent == Intent.OUT_OF_SCOPE:
        # 短路：不做 embedding、不查图、不调 LLM
        sessions.commit_turn(
            state, req.message, OUT_OF_SCOPE_REPLY, intent_res.intent, [], []
        )
        latency["total"] = (time.perf_counter() - t_start) * 1000
        return ChatResponse(
            session_id=req.session_id,
            answer=OUT_OF_SCOPE_REPLY,
            intent=intent_res,
            followup=FollowupResult(),
            latency_ms=latency,
        )

    # ---------------- 2. 追问识别 ----------------
    t0 = time.perf_counter()
    followup, focus = _detect_followup(sessions, state, req.message)
    latency["followup"] = (time.perf_counter() - t0) * 1000

    # ---------------- 3. 检索 ----------------
    t0 = time.perf_counter()
    seed_preview = [
        e.entity.entity_id
        for e in get_retriever().linker.link(req.message, top_k=settings.entity_link_top_k)
    ]
    # 新话题不做检索去重：上一个话题扩展过的实体，和这一轮没关系
    prune = sessions.prune_entity_ids(state, seed_preview) if followup.is_followup else set()
    retrieval = get_retriever().retrieve(
        req.message, mode=req.retrieval_mode, exclude_entity_ids=prune
    )
    latency["retrieval"] = (time.perf_counter() - t0) * 1000

    # ---------------- 4. 生成 ----------------
    history = sessions.build_context(state, followup=followup)
    current_plan = _plan_to_text(state.working_plan) if followup.is_followup else ""
    use_agent = req.use_agent if req.use_agent is not None else settings.agent_enabled

    t0 = time.perf_counter()
    agent_trace = None
    if use_agent and intent_res.intent in AGENT_INTENTS:
        agent_trace = get_agent().run(
            question=req.message,
            intent=intent_res.intent,
            context=retrieval.context_text(),
            history=history,
            current_plan=current_plan,
        )
        answer = agent_trace.final_answer
    else:
        messages = build_prompt(
            intent_res.intent, req.message, retrieval.context_text(), history, current_plan
        )
        answer = get_llm().chat(messages, purpose=f"gen_{intent_res.intent.value}")
    latency["generation"] = (time.perf_counter() - t0) * 1000

    # ---------------- 5. 规则兜底校验 ----------------
    t0 = time.perf_counter()
    plan: ConfigPlan | None = None
    validation = None
    if intent_res.intent in PLAN_INTENTS:
        plan = extract_plan(answer, use_llm_fallback=False)
        if plan.values:
            validation, _ = validate_with_base(plan, detect_scope(req.message))
            answer = _append_validation(answer, validation)
    latency["validation"] = (time.perf_counter() - t0) * 1000

    # ---------------- 6. 回写会话 ----------------
    entity_ids = [e.entity.entity_id for e in retrieval.subgraph_entities]
    item_keys = [
        e.entity.props.get("key")
        for e in retrieval.subgraph_entities
        if e.entity.props.get("key")
    ]
    sessions.commit_turn(
        state, req.message, answer, intent_res.intent, entity_ids, item_keys, plan,
        focus_entity_ids=focus, is_followup=followup.is_followup,
    )

    latency["total"] = (time.perf_counter() - t_start) * 1000
    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        intent=intent_res,
        retrieval=retrieval,
        agent_trace=agent_trace,
        validation=validation,
        plan=plan,
        followup=followup,
        latency_ms=latency,
    )


# =============================================================================
# 流式
# =============================================================================


def handle_chat_stream(req: ChatRequest) -> Iterator[dict]:
    """SSE 事件流。

    Agent 路径本质是多步的，没法把中间的工具调用伪装成流式 token，
    所以这里的做法是：**把每一步的思考和工具调用作为独立事件推给前端**。
    用户看到的是"正在查询 XX 配置的依赖链…"这样的进度，
    比一个转圈的 loading 强得多，也让 Agent 的推理过程对用户可解释。
    """
    sessions = get_session_manager()
    state = sessions.load(req.session_id)

    intent_res = (
        IntentResult(intent=req.force_intent, confidence=1.0)
        if req.force_intent is not None
        else get_classifier().classify(req.message)
    )
    yield {"event": "intent", "data": intent_res.model_dump(mode="json")}

    if intent_res.intent == Intent.OUT_OF_SCOPE:
        yield {"event": "token", "data": {"text": OUT_OF_SCOPE_REPLY}}
        sessions.commit_turn(state, req.message, OUT_OF_SCOPE_REPLY, intent_res.intent, [], [])
        yield {"event": "done", "data": {"reason": "out_of_scope"}}
        return

    followup, focus = _detect_followup(sessions, state, req.message)
    yield {"event": "followup", "data": followup.model_dump(mode="json")}

    seed_preview = [e.entity.entity_id for e in get_retriever().linker.link(req.message)]
    prune = sessions.prune_entity_ids(state, seed_preview) if followup.is_followup else set()
    retrieval = get_retriever().retrieve(
        req.message, mode=req.retrieval_mode, exclude_entity_ids=prune
    )
    yield {"event": "retrieval", "data": _retrieval_summary(retrieval)}

    history = sessions.build_context(state, followup=followup)
    current_plan = _plan_to_text(state.working_plan) if followup.is_followup else ""
    use_agent = req.use_agent if req.use_agent is not None else settings.agent_enabled

    if use_agent and intent_res.intent in AGENT_INTENTS:
        trace = get_agent().run(
            question=req.message, intent=intent_res.intent,
            context=retrieval.context_text(), history=history, current_plan=current_plan,
        )
        for step in trace.steps:
            yield {
                "event": "agent_step",
                "data": {
                    "step": step.step,
                    "thought": step.thought,
                    "tool": step.tool_call.name if step.tool_call else None,
                    "arguments": step.tool_call.arguments if step.tool_call else {},
                    "ok": step.tool_result.ok if step.tool_result else None,
                    "observation": (step.tool_result.content[:600] if step.tool_result else ""),
                    "latency_ms": round(step.tool_result.latency_ms, 1) if step.tool_result else 0,
                },
            }
        answer = trace.final_answer
        yield {"event": "token", "data": {"text": answer}}
    else:
        messages = build_prompt(
            intent_res.intent, req.message, retrieval.context_text(), history, current_plan
        )
        chunks: list[str] = []
        for delta in get_llm().chat_stream(messages, purpose=f"gen_{intent_res.intent.value}"):
            chunks.append(delta)
            yield {"event": "token", "data": {"text": delta}}
        answer = "".join(chunks)

    plan = None
    if intent_res.intent in PLAN_INTENTS:
        plan = extract_plan(answer, use_llm_fallback=False)
        if plan.values:
            report, _ = validate_with_base(plan, detect_scope(req.message))
            suffix = _validation_suffix(report)
            if suffix:
                yield {"event": "token", "data": {"text": suffix}}
                answer += suffix
            yield {"event": "validation", "data": report.model_dump(mode="json")}

    entity_ids = [e.entity.entity_id for e in retrieval.subgraph_entities]
    item_keys = [
        e.entity.props.get("key") for e in retrieval.subgraph_entities
        if e.entity.props.get("key")
    ]
    sessions.commit_turn(
        state, req.message, answer, intent_res.intent, entity_ids, item_keys, plan,
        focus_entity_ids=focus, is_followup=followup.is_followup,
    )
    yield {"event": "done", "data": {"reason": "final"}}


# =============================================================================
# 辅助
# =============================================================================


def _detect_followup(sessions, state, message: str):
    """追问识别：本轮原样提到了哪些实体 + 它们和上一轮焦点在图上是不是邻居。"""
    mentioned = get_retriever().linker.mentions(message)
    return sessions.detect_followup(state, message, mentioned, _graph_neighbors)


def _graph_neighbors(entity_ids: list[str]) -> set[str]:
    """图上一跳邻居，只要配置项和规则（分类、维度这类节点人人相连，算进来等于没筛）。"""
    found, _rels = get_graph_store().expand(entity_ids, hops=1, max_nodes=30)
    return {eid for eid in found if eid.startswith(("config:", "rule:"))}


def _plan_to_text(plan: ConfigPlan) -> str:
    if not plan.values:
        return ""
    lines = [f"场景：{plan.scenario}"] if plan.scenario else []
    lines.extend(
        f"- {k} = {v}" + (f"（{plan.dimensions[k]} 维度）" if k in plan.dimensions else "")
        for k, v in plan.values.items()
    )
    return "\n".join(lines)


def _validation_suffix(report) -> str:
    if report.passed and not report.warns:
        return "\n\n---\n✅ **规则引擎校验通过**：依赖链完整、无互斥冲突、组合规则(R001~R004)全部满足。"
    if not report.passed:
        # Agent 已经被退回重改过仍不合规：明确拦截，不让编辑把它当成可用方案
        return ("\n\n---\n### ⛔ 方案未通过规则校验，请勿直接使用\n\n" + report.to_text()
                + "\n\n> 以上由确定性规则引擎校验得出，优先级高于上文的模型判断。")
    return "\n\n---\n### ⚠️ 规则引擎校验结果\n\n" + report.to_text()


def _append_validation(answer: str, report) -> str:
    return answer + _validation_suffix(report)


def _retrieval_summary(r: RetrievalResult) -> dict:
    return {
        "seed_entities": [
            {"id": e.entity.entity_id, "name": e.entity.name, "score": round(e.score, 3)}
            for e in r.seed_entities
        ],
        "subgraph_size": len(r.subgraph_entities),
        "expanded": [
            {"name": e.entity.name, "hop": e.hop, "score": round(e.score, 3)}
            for e in r.subgraph_entities if e.hit_type == "expanded"
        ][:15],
        "triples": [t.to_text() for t in r.triples][:25],
        "chunks": [
            {
                "id": c.chunk.chunk_id,
                "doc": c.chunk.doc_title,
                "section": c.chunk.section,
                "score": round(c.score, 3),
                "source": c.source,
                "preview": c.chunk.text[:120],
            }
            for c in r.chunks
        ],
        "pruned_count": len(r.pruned_entity_ids),
        "latency_ms": {k: round(v, 1) for k, v in r.latency_ms.items()},
    }
