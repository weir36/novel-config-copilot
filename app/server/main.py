"""FastAPI 服务入口。"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.chat.pipeline import handle_chat, handle_chat_stream
from app.foundation.registry import get_registry
from app.foundation.settings import settings
from app.retrieval.embedder import embedder_meta
from app.generation.llm_client import STATS, llm_meta
from app.session.session import get_session_manager
from app.retrieval.graphrag import get_retriever
from app.validation.rule_engine import get_rule_engine
from app.foundation.models import ChatRequest, ChatResponse, ConfigPlan, ValidationReport

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
logger = logging.getLogger("lcc")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时把索引全部加载并打印实际选中的后端 —— 降级是静默发生的，
    # 不打出来的话很容易出现"以为在用 Neo4j，其实跑的是内存图"这种误判
    index_ok = (settings.index_dir / "chunks_vectors.npy").exists()
    if not index_ok:
        logger.warning("未检测到索引文件，请先执行：python -m app.indexing.build")
    else:
        r = get_retriever()
        # 意图分类器要编码全部种子样本，首次调用要几百毫秒。
        # 放到启动时预热，否则第一个用户的请求会白白多等这一下。
        from app.intent.classifier import get_classifier  # noqa: PLC0415

        get_classifier().classify("预热")
        logger.info(
            "后端就绪 | 向量=%s 关键词=%s 图谱=%s 会话=%s embedding=%s LLM=%s",
            r.chunk_vs.backend, r.chunk_kw.backend, r.graph.backend,
            get_session_manager().store.backend,
            embedder_meta()["backend"], llm_meta()["provider"],
        )
    yield


app = FastAPI(title="网文作品资源配置助手", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# =============================================================================
# 对话
# =============================================================================


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return handle_chat(req)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    def gen():
        try:
            for evt in handle_chat_stream(req):
                yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001 - SSE 里异常必须转成事件，否则前端只会看到连接断开
            logger.exception("流式处理异常")
            payload = json.dumps({"message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str) -> dict:
    get_session_manager().reset(session_id)
    return {"ok": True}


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    return get_session_manager().load(session_id).model_dump(mode="json")


# =============================================================================
# 配置域只读接口（前端侧边栏 / 调试用）
# =============================================================================


@app.get("/api/config/items")
def list_items(category: str | None = None) -> list[dict]:
    reg = get_registry()
    items = reg.items_by_category(category) if category else list(reg.items.values())
    return [
        {
            "key": i.key, "name": i.name, "category": i.category, "type": i.type,
            "default": i.default, "options": i.options, "range": i.range,
            "dimensions": i.dimensions, "desc": i.desc,
            "depends_on": i.depends_on, "mutex_with": i.mutex_with,
        }
        for i in items
    ]


@app.get("/api/config/items/{key}")
def get_item(key: str) -> dict:
    item = get_registry().resolve(key)
    if item is None:
        raise HTTPException(404, f"配置项 {key} 不存在")
    reg = get_registry()
    return {
        "key": item.key, "name": item.name, "desc": item.desc, "type": item.type,
        "options": item.options, "range": item.range, "default": item.default,
        "dimensions": item.dimensions,
        "dependency_chain": reg.dependency_chain(item.key),
        "mutex_with": item.mutex_with,
        "requires_value": item.requires_value,
        "enhances": [reg.effects.get(e, e) for e in item.enhances],
        "weakens": [reg.effects.get(e, e) for e in item.weakens],
    }


@app.post("/api/config/validate", response_model=ValidationReport)
def validate(plan: ConfigPlan) -> ValidationReport:
    return get_rule_engine().validate(plan)


@app.post("/api/config/autofix")
def autofix(plan: ConfigPlan) -> dict:
    fixed, notes = get_rule_engine().auto_fix(plan)
    report = get_rule_engine().validate(fixed)
    return {"plan": fixed.model_dump(mode="json"), "notes": notes,
            "report": report.model_dump(mode="json")}


@app.get("/api/graph/neighbors/{entity_id:path}")
def graph_neighbors(entity_id: str, hops: int = 1) -> dict:
    """给前端画关系图用。"""
    from app.retrieval.graph_store import relations_to_triples  # noqa: PLC0415

    graph = get_retriever().graph
    visited, relations = graph.expand([entity_id], hops=hops, max_nodes=60)
    ent_map = {eid: e for eid, (e, _h, _w) in visited.items()}
    return {
        "nodes": [
            {"id": eid, "name": e.name, "type": e.type.value, "hop": hop}
            for eid, (e, hop, _w) in visited.items()
        ],
        "edges": [
            {"source": r.source_id, "target": r.target_id, "type": r.type.value}
            for r in relations
        ],
        "triples": [t.to_text() for t in relations_to_triples(relations, ent_map)],
    }


# =============================================================================
# 运维
# =============================================================================


@app.get("/api/health")
def health() -> dict:
    r = get_retriever()
    return {
        "status": "ok",
        "backends": {
            "vector": r.chunk_vs.backend,
            "keyword": r.chunk_kw.backend,
            "graph": r.graph.backend,
            "session": get_session_manager().store.backend,
            "embedding": embedder_meta(),
            "llm": llm_meta(),
        },
        "index": {
            "chunks": len(r.chunk_vs.ids),
            "entities": len(r.linker.store.ids),
            "config_items": len(get_registry().items),
        },
    }


@app.get("/api/metrics")
def metrics() -> dict:
    return {"llm": STATS.snapshot()}


@app.post("/api/admin/reload")
def reload_index() -> dict:
    """离线建库完成后热加载新索引，不用重启服务。"""
    from app.intent.classifier import reset_classifier  # noqa: PLC0415

    get_retriever().refresh()
    reset_classifier()
    return health()


# =============================================================================
# 前端
# =============================================================================


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
def appjs() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/style.css")
def stylecss() -> FileResponse:
    return FileResponse(WEB_DIR / "style.css", media_type="text/css")
