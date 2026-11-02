"""会话状态管理（上下文缓存）。

存的不是聊天记录，是**结构化的会话状态**：每轮的问题、回答摘要、命中实体集合、
以及用户当前正在编辑的配置方案。这三样东西分别服务三个目的：

  1. 多轮连贯 —— "那保底呢" 这种指代必须靠上一轮的实体集合才能解析。
     但不是每一轮都该带历史：先做**追问识别**，判断这一句是不是接着上文问的，
     是追问才挑出相关的几轮带上，是新话题就不带，免得上一个话题干扰这一轮；
  2. **检索剪枝** —— 会话级累积的 seen_entity_ids 传给 GraphRAG，
     已经扩展过的外围实体这一轮不再重复扩展。多轮对话里这能省掉 30%~50% 的
     子图节点，直接反映在上下文长度和 token 成本上；
  3. 方案增量编辑 —— working_plan 让"在现有配置上改"这类请求有明确的基准，
     而不是每轮让 LLM 从对话历史里重新猜当前配置是什么。

存储后端 Redis 优先（多实例共享、天然 TTL），探测不到降级到进程内 LRU。
"""
from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable

from app.foundation.settings import settings
from app.foundation.models import ConfigPlan, FollowupResult, Intent, SessionState, Turn

logger = logging.getLogger(__name__)


class BaseSessionStore:
    backend = "base"

    def load(self, session_id: str) -> SessionState: ...
    def save(self, state: SessionState) -> None: ...
    def delete(self, session_id: str) -> None: ...


class InProcSessionStore(BaseSessionStore):
    backend = "inproc"

    def __init__(self, capacity: int = 1000) -> None:
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.capacity = capacity

    def load(self, session_id: str) -> SessionState:
        item = self._data.get(session_id)
        if item is None:
            return SessionState(session_id=session_id)
        ts, payload = item
        if time.time() - ts > settings.redis_ttl_s:
            self._data.pop(session_id, None)
            return SessionState(session_id=session_id)
        self._data.move_to_end(session_id)
        return SessionState.model_validate_json(payload)

    def save(self, state: SessionState) -> None:
        self._data[state.session_id] = (time.time(), state.model_dump_json())
        self._data.move_to_end(state.session_id)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionStore(BaseSessionStore):
    backend = "redis"

    def __init__(self) -> None:
        import redis  # noqa: PLC0415

        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.client.ping()

    @staticmethod
    def _key(session_id: str) -> str:
        return f"ncc:session:{session_id}"  # 和项目一（lcc:）分开，共用一个 Redis 也不会串

    def load(self, session_id: str) -> SessionState:
        raw = self.client.get(self._key(session_id))
        if not raw:
            return SessionState(session_id=session_id)
        try:
            return SessionState.model_validate_json(raw)
        except ValueError:
            # schema 演进后老数据反序列化失败：丢弃重建，不要让历史数据打挂新版本
            logger.warning("会话 %s 反序列化失败，已重置", session_id)
            return SessionState(session_id=session_id)

    def save(self, state: SessionState) -> None:
        self.client.setex(
            self._key(state.session_id), settings.redis_ttl_s, state.model_dump_json()
        )

    def delete(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))


class SessionManager:
    def __init__(self) -> None:
        self.store = _make_store()

    # ---------------- 读写 ----------------
    def load(self, session_id: str) -> SessionState:
        return self.store.load(session_id)

    def save(self, state: SessionState) -> None:
        # 只保留最近 N 轮明细，更早的已经压进 rolling_summary
        if len(state.turns) > settings.max_history_turns:
            state.turns = state.turns[-settings.max_history_turns :]
        self.store.save(state)

    def reset(self, session_id: str) -> None:
        self.store.delete(session_id)

    # ---------------- 一轮对话的收尾 ----------------
    def commit_turn(
        self,
        state: SessionState,
        question: str,
        answer: str,
        intent,
        entity_ids: list[str],
        item_keys: list[str],
        plan: ConfigPlan | None = None,
        focus_entity_ids: list[str] | None = None,
        is_followup: bool = False,
    ) -> None:
        turn = Turn(
            turn_id=(state.turns[-1].turn_id if state.turns else 0) + 1,
            question=question,
            answer=answer,
            intent=intent,
            entity_ids=entity_ids,
            item_keys=item_keys,
            focus_entity_ids=focus_entity_ids or [],
            is_followup=is_followup,
            summary=_compress_turn(question, answer),
        )
        state.turns.append(turn)

        merged = list(state.seen_entity_ids)
        merged.extend(e for e in entity_ids if e not in merged)
        state.seen_entity_ids = merged[-120:]

        if plan is not None and plan.values:
            # 增量合并而不是整体替换：编辑说"再加个全勤奖"时，
            # 不该把上一轮已经定好的十几项冲掉
            state.working_plan.values.update(plan.values)
            state.working_plan.dimensions.update(plan.dimensions)
            state.working_plan.rationale.update(plan.rationale)
            if plan.scenario:
                state.working_plan.scenario = plan.scenario

        if len(state.turns) > settings.summarize_after_turns:
            state.rolling_summary = _roll_summary(state)

        self.save(state)

    # ---------------- 追问识别 ----------------
    def detect_followup(
        self,
        state: SessionState,
        question: str,
        mentioned_ids: list[str],
        neighbors_of: Callable[[list[str]], set[str]] | None = None,
    ) -> tuple[FollowupResult, list[str]]:
        """判断这一句是不是接着上文问的。返回 (识别结果, 本轮话题焦点)。

        三个信号，命中任意一个就算追问：
          1. 指代 / 承接词：以"那""再""换成"开头，或出现"这个""它""刚才"，或以"呢"结尾；
          2. 配置项重合：本轮提到的配置项，和上一轮焦点或焦点的一跳邻居有交集
             （问完全勤奖接着问日更字数，两者在图上直接相连）；
          3. 在改上一轮的方案：上一轮出了方案，这一轮出现"改到""去掉""加上"这类改动词；
          4. 短问且没提任何配置项：比如"上限是多少"，不接上文根本没法回答。

        用规则不用模型：这一步每轮都跑，规则零成本、可解释，判错了也看得出是哪条信号误触。
        """
        q = question.strip()
        cur = list(dict.fromkeys(mentioned_ids))
        if not state.turns:
            return FollowupResult(), cur

        prev = state.turns[-1]
        reasons: list[str] = []

        if q.startswith(_FOLLOWUP_STARTS) or any(m in q for m in _FOLLOWUP_REFS) or q.endswith("呢"):
            reasons.append("指代或承接词")

        prev_focus = set(prev.focus_entity_ids)
        related = set(prev_focus)
        if neighbors_of and prev_focus:
            related |= neighbors_of(sorted(prev_focus))
        if set(cur) & related:
            reasons.append("配置项和上一轮重合")

        if prev.intent in (Intent.RECOMMEND, Intent.OPTIMIZE) and any(w in q for w in _PLAN_EDIT_WORDS):
            reasons.append("在改上一轮的方案")

        if not cur and len(q) <= 12:
            reasons.append("短问且没提配置项")

        if not reasons:
            return FollowupResult(), cur

        # 追问：焦点继承上一轮，相关轮次 = 上一轮 + 焦点有交集的更早轮次
        focus = list(dict.fromkeys([*cur, *prev.focus_entity_ids]))
        focus_set = set(focus)
        related_ids = [
            t.turn_id for t in state.turns
            if t is prev or focus_set & set(t.focus_entity_ids)
        ]
        return FollowupResult(is_followup=True, reasons=reasons, related_turn_ids=related_ids), focus

    # ---------------- 给 LLM 的精简上下文 ----------------
    def build_context(
        self, state: SessionState, max_chars: int = 1200, followup: FollowupResult | None = None
    ) -> str:
        """拼历史上下文。

        传了 followup 时按追问识别的结果来：新话题不带历史，追问只带相关的几轮。
        不传时沿用旧行为（最近 3 轮 + 更早的摘要），保持对旧调用方的兼容。
        """
        if followup is not None and not followup.is_followup:
            return ""
        parts: list[str] = []
        if followup is not None:
            wanted = set(followup.related_turn_ids)
            turns = [t for t in state.turns if t.turn_id in wanted][-4:]
        else:
            if state.rolling_summary:
                parts.append(f"【更早对话摘要】{state.rolling_summary}")
            turns = state.turns[-3:]
        for t in turns:
            parts.append(f"[第{t.turn_id}轮] 编辑：{t.question}\n助手：{t.summary}")
        if state.working_plan.values:
            kv = "、".join(f"{k}={v}" for k, v in list(state.working_plan.values.items())[:20])
            parts.append(f"【当前工作区配置方案】{kv}")
        text = "\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text

    def prune_entity_ids(self, state: SessionState, current_seed_ids: list[str]) -> set[str]:
        """给 GraphRAG 的剪枝集合：历史见过、但本轮没被直接命中的实体。

        本轮命中的种子必须保留 —— 用户又提到它，说明它仍是焦点。
        """
        return set(state.seen_entity_ids) - set(current_seed_ids)


_FOLLOWUP_STARTS = ("那", "再", "换成", "还有", "另外")
_FOLLOWUP_REFS = ("这个", "它", "刚才", "上面", "上一个", "同样", "跟它", "和它")
_PLAN_EDIT_WORDS = ("改到", "改成", "调到", "去掉", "删掉", "加上", "加个")


def _compress_turn(question: str, answer: str, limit: int = 160) -> str:
    """单轮回答压缩。用规则而不是 LLM：这一步在每轮都跑，
    用 LLM 意味着每轮多一次调用和几百毫秒延迟，不值当。"""
    text = " ".join(answer.split())
    if len(text) <= limit:
        return text
    head = text[: limit - 40]
    tail = text[-30:]
    return f"{head}……{tail}"


def _roll_summary(state: SessionState) -> str:
    """滚动摘要：把早于最近 N 轮的内容压成一句话。同样走规则。"""
    old = state.turns[: -settings.summarize_after_turns]
    if not old:
        return state.rolling_summary
    topics: list[str] = []
    for t in old:
        if t.item_keys:
            topics.extend(t.item_keys)
    uniq = list(dict.fromkeys(topics))[:12]
    intents = list(dict.fromkeys(t.intent.value for t in old))
    return f"共 {len(old)} 轮，涉及意图 {'/'.join(intents)}，围绕配置项：{'、'.join(uniq) or '无'}"


def _make_store() -> BaseSessionStore:
    choice = settings.memory_backend
    if choice in ("auto", "redis"):
        try:
            store = RedisSessionStore()
            logger.info("SessionStore: redis @ %s", settings.redis_url)
            return store
        except Exception as e:  # noqa: BLE001
            if choice == "redis":
                raise
            logger.warning("Redis 不可用（%s），会话降级到进程内存储", type(e).__name__)
    return InProcSessionStore()


_MANAGER: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = SessionManager()
    return _MANAGER
