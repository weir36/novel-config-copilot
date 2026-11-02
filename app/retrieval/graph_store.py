"""知识图谱存储与子图扩展。

对外只暴露三个能力：写入图、按实体 ID 做 N 跳扩展、把子图导出成三元组文本。
底下有两套实现：

  - Neo4jGraphStore：生产形态，扩展逻辑用一条变长路径 Cypher 完成，
    关系类型和跳数都参数化，图规模上去之后不需要把边捞回内存。
  - NetworkXGraphStore：本地形态，同样的 BFS 语义，无外部依赖。

两套实现共用同一套「带权扩展」策略：不同关系类型对相关性的贡献不同 ——
依赖/互斥是硬约束必须跟到底，增强/削弱这类效果关系跟一跳就够，
再往下就开始引入噪声了。这个权重直接决定第二阶段召回的信噪比。
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import networkx as nx

from app.foundation.settings import settings
from app.foundation.models import Entity, EntityType, KnowledgeGraph, Relation, RelationType, Triple

logger = logging.getLogger(__name__)

# 关系类型 -> (扩展衰减系数, 最大跟随跳数)
RELATION_POLICY: dict[str, tuple[float, int]] = {
    RelationType.DEPENDS_ON.value: (0.95, 2),
    RelationType.REQUIRES_VALUE.value: (0.95, 2),
    RelationType.MUTEX_WITH.value: (0.95, 2),
    RelationType.CONSTRAINED_BY.value: (0.90, 2),
    RelationType.ENHANCES.value: (0.55, 1),
    RelationType.WEAKENS.value: (0.55, 1),
    RelationType.BELONGS_TO.value: (0.40, 1),
    RelationType.APPLIES_TO.value: (0.40, 1),
    RelationType.USED_IN.value: (0.60, 1),
}

PREDICATE_ZH: dict[str, str] = {
    "DEPENDS_ON": "依赖",
    "REQUIRES_VALUE": "要求取值为",
    "MUTEX_WITH": "互斥于",
    "ENHANCES": "增强",
    "WEAKENS": "削弱",
    "APPLIES_TO": "可作用于维度",
    "BELONGS_TO": "属于分类",
    "CONSTRAINED_BY": "受规则约束",
    "USED_IN": "被方案使用",
}

# 无向对待的关系（互斥天然双向）
UNDIRECTED = {RelationType.MUTEX_WITH.value}


class BaseGraphStore:
    backend = "base"

    def write(self, kg: KnowledgeGraph) -> None: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def expand(
        self, seed_ids: list[str], hops: int, max_nodes: int, exclude_ids: set[str] | None = None
    ) -> tuple[dict[str, tuple[Entity, int, float]], list[Relation]]: ...


# =============================================================================
# NetworkX
# =============================================================================


class NetworkXGraphStore(BaseGraphStore):
    backend = "networkx"

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.entities: dict[str, Entity] = {}

    def write(self, kg: KnowledgeGraph) -> None:
        self.g = nx.MultiDiGraph()
        self.entities = {e.entity_id: e for e in kg.entities}
        for e in kg.entities:
            self.g.add_node(e.entity_id, type=e.type.value, name=e.name)
        for r in kg.relations:
            if r.source_id not in self.entities or r.target_id not in self.entities:
                continue
            self.g.add_edge(r.source_id, r.target_id, key=r.type.value, rel=r)
        logger.info(
            "NetworkXGraphStore: %d nodes / %d edges",
            self.g.number_of_nodes(), self.g.number_of_edges(),
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def _neighbors(self, node: str) -> list[tuple[str, Relation]]:
        out: list[tuple[str, Relation]] = []
        for _s, t, data in self.g.out_edges(node, data=True):
            out.append((t, data["rel"]))
        for s, _t, data in self.g.in_edges(node, data=True):
            rel: Relation = data["rel"]
            # 互斥是双向的；其余关系反向也有检索价值（"谁依赖我"同样重要），
            # 但反向跟随要打折，避免从一个热点实体炸开整张图
            out.append((s, rel))
        return out

    def expand(self, seed_ids, hops, max_nodes, exclude_ids=None):
        exclude = exclude_ids or set()
        visited: dict[str, tuple[Entity, int, float]] = {}
        edges: dict[tuple[str, str, str], Relation] = {}
        q: deque[tuple[str, int, float]] = deque()

        for sid in seed_ids:
            ent = self.entities.get(sid)
            if ent is None:
                continue
            visited[sid] = (ent, 0, 1.0)
            q.append((sid, 0, 1.0))

        while q and len(visited) < max_nodes:
            node, hop, weight = q.popleft()
            if hop >= hops:
                continue
            for nbr, rel in self._neighbors(node):
                policy = RELATION_POLICY.get(rel.type.value, (0.3, 1))
                decay, max_hop = policy
                if hop + 1 > max_hop:
                    continue
                # 反向跟随（我不是这条边的 source）额外打 0.8 折
                is_reverse = rel.source_id != node and rel.type.value not in UNDIRECTED
                new_w = weight * decay * (0.8 if is_reverse else 1.0)
                ekey = (rel.source_id, rel.target_id, rel.type.value)
                edges.setdefault(ekey, rel)
                if nbr in exclude:
                    continue
                ent = self.entities.get(nbr)
                if ent is None:
                    continue
                if nbr not in visited or visited[nbr][2] < new_w:
                    visited[nbr] = (ent, hop + 1, new_w)
                    q.append((nbr, hop + 1, new_w))
                if len(visited) >= max_nodes:
                    break

        return visited, list(edges.values())


# =============================================================================
# Neo4j
# =============================================================================


class Neo4jGraphStore(BaseGraphStore):
    backend = "neo4j"

    def __init__(self) -> None:
        from neo4j import GraphDatabase  # noqa: PLC0415

        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self.driver.verify_connectivity()
        self._cache: dict[str, Entity] = {}

    def close(self) -> None:
        self.driver.close()

    def write(self, kg: KnowledgeGraph) -> None:
        with self.driver.session(database=settings.neo4j_database) as s:
            s.run("MATCH (n:KGNode) DETACH DELETE n")
            s.run(
                "CREATE CONSTRAINT kgnode_id IF NOT EXISTS "
                "FOR (n:KGNode) REQUIRE n.entity_id IS UNIQUE"
            )
            s.run(
                """
                UNWIND $rows AS row
                MERGE (n:KGNode {entity_id: row.entity_id})
                SET n.name = row.name, n.type = row.type,
                    n.aliases = row.aliases, n.description = row.description,
                    n.props = row.props
                """,
                rows=[
                    {
                        "entity_id": e.entity_id,
                        "name": e.name,
                        "type": e.type.value,
                        "aliases": e.aliases,
                        "description": e.description,
                        "props": json.dumps(e.props, ensure_ascii=False),
                    }
                    for e in kg.entities
                ],
            )
            # 关系类型无法参数化，按类型分组各跑一条
            by_type: dict[str, list[Relation]] = {}
            for r in kg.relations:
                by_type.setdefault(r.type.value, []).append(r)
            for rtype, rels in by_type.items():
                s.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:KGNode {{entity_id: row.src}})
                    MATCH (b:KGNode {{entity_id: row.dst}})
                    MERGE (a)-[r:{rtype}]->(b)
                    SET r.props = row.props, r.confidence = row.confidence
                    """,
                    rows=[
                        {
                            "src": r.source_id,
                            "dst": r.target_id,
                            "props": json.dumps(r.props, ensure_ascii=False),
                            "confidence": r.confidence,
                        }
                        for r in rels
                    ],
                )
        self._cache = {e.entity_id: e for e in kg.entities}
        logger.info("Neo4jGraphStore: %d nodes / %d rels", len(kg.entities), len(kg.relations))

    def get_entity(self, entity_id: str) -> Entity | None:
        if entity_id in self._cache:
            return self._cache[entity_id]
        with self.driver.session(database=settings.neo4j_database) as s:
            rec = s.run(
                "MATCH (n:KGNode {entity_id: $eid}) RETURN n", eid=entity_id
            ).single()
        if rec is None:
            return None
        ent = _node_to_entity(rec["n"])
        self._cache[entity_id] = ent
        return ent

    def expand(self, seed_ids, hops, max_nodes, exclude_ids=None):
        exclude = list(exclude_ids or set())
        # 变长路径一次捞回：APOC 不一定可用，所以用原生变长模式 *1..N。
        # N 用字符串插值是因为 Cypher 不允许把跳数做成参数，seed 与其余全部参数化。
        cypher = f"""
        MATCH (s:KGNode) WHERE s.entity_id IN $seeds
        MATCH p = (s)-[*1..{int(hops)}]-(m:KGNode)
        WHERE NOT m.entity_id IN $exclude
        // 驱动返回的关系对象里，start_node / end_node 是不带属性的占位节点，
        // 拿不到 entity_id，所以两端 id 和关系属性在 Cypher 里直接取出来
        WITH m, length(p) AS hop,
             [r IN relationships(p) | {{type: type(r), src: startNode(r).entity_id,
                dst: endNode(r).entity_id, props: r.props, confidence: r.confidence}}] AS rels
        ORDER BY hop ASC
        WITH m, head(collect({{rels: rels, hop: hop}})) AS best
        RETURN m, best.hop AS hop, best.rels AS rels
        LIMIT $limit
        """
        visited: dict[str, tuple[Entity, int, float]] = {}
        edges: dict[tuple[str, str, str], Relation] = {}

        with self.driver.session(database=settings.neo4j_database) as s:
            for sid in seed_ids:
                ent = self.get_entity(sid)
                if ent is not None:
                    visited[sid] = (ent, 0, 1.0)
            for rec in s.run(cypher, seeds=seed_ids, exclude=exclude, limit=max_nodes):
                ent = _node_to_entity(rec["m"])
                hop = int(rec["hop"])
                weight = 1.0
                ok = True
                for rel in rec["rels"]:
                    decay, max_hop = RELATION_POLICY.get(rel["type"], (0.3, 1))
                    if hop > max_hop:
                        ok = False
                        break
                    weight *= decay
                    key = (rel["src"], rel["dst"], rel["type"])
                    edges.setdefault(key, _rel_to_relation(rel))
                if not ok:
                    continue
                if ent.entity_id not in visited or visited[ent.entity_id][2] < weight:
                    visited[ent.entity_id] = (ent, hop, weight)
        return visited, list(edges.values())


def _node_to_entity(node) -> Entity:
    props = node.get("props") or "{}"
    return Entity(
        entity_id=node["entity_id"],
        name=node["name"],
        type=EntityType(node["type"]),
        aliases=list(node.get("aliases") or []),
        description=node.get("description") or "",
        props=json.loads(props) if isinstance(props, str) else props,
    )


def _rel_to_relation(rel: dict) -> Relation:
    props = rel.get("props") or "{}"
    return Relation(
        source_id=rel["src"],
        target_id=rel["dst"],
        type=RelationType(rel["type"]),
        props=json.loads(props) if isinstance(props, str) else props,
        confidence=float(rel.get("confidence") or 1.0),
    )


# =============================================================================
# 三元组导出 & 工厂
# =============================================================================


def relations_to_triples(
    relations: list[Relation], entities: dict[str, Entity]
) -> list[Triple]:
    """把子图导成给 LLM 读的自然语言三元组。

    这一步是 GraphRAG 相对朴素 RAG 的核心增量：文本段里"加更活动依赖全勤奖"这句话
    可能和"全勤奖依赖日更字数要求"分散在不同段落里，拼不到一起；
    但在图上它们是一条连通路径，导出成三元组后 LLM 一眼就能看到完整依赖链。
    """
    triples: list[Triple] = []
    for r in relations:
        s = entities.get(r.source_id)
        t = entities.get(r.target_id)
        if s is None or t is None:
            continue
        note = ""
        if r.type == RelationType.REQUIRES_VALUE and "value" in r.props:
            note = f"（取值须为 {r.props['value']}）"
        triples.append(
            Triple(
                subject=s.name,
                predicate=PREDICATE_ZH.get(r.type.value, r.type.value),
                object=t.name,
                note=note,
            )
        )
    return triples


_GRAPH: BaseGraphStore | None = None
KG_JSON_PATH = settings.index_dir / "knowledge_graph.json"


def get_graph_store() -> BaseGraphStore:
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    choice = settings.graph_backend
    store: BaseGraphStore | None = None
    if choice in ("auto", "neo4j"):
        try:
            store = Neo4jGraphStore()
        except Exception as e:  # noqa: BLE001
            if choice == "neo4j":
                raise
            logger.warning("Neo4j 不可用（%s），降级到 networkx 内存图", type(e).__name__)
    if store is None:
        store = NetworkXGraphStore()
        if KG_JSON_PATH.exists():
            kg = KnowledgeGraph.model_validate_json(KG_JSON_PATH.read_text(encoding="utf-8"))
            store.write(kg)
    _GRAPH = store
    return store


def reset_graph() -> None:
    global _GRAPH
    _GRAPH = None
