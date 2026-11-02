"""LLM 客户端。

统一走 OpenAI-compatible 协议，因此换模型只改 .env（豆包/方舟、DeepSeek、Qwen、
自建 vLLM 全都兼容）。除真实客户端外还有一个 MockLLM：
没配 API key 时自动启用，用确定性模板应答，让整条链路（含评测）在离线环境下也能跑通。

三个约定：
  - chat_json 负责把模型输出掰成合法 JSON：先剥 ```json 围栏，再做括号配平截取，
    仍失败则带着报错重试一次。线上 JSON 解析失败是常态，不处理就是随机 500。
  - 重试只对网络错误和 5xx 生效，4xx 直接抛 —— 参数错了重试多少次都是错。
  - 所有调用打点 token 与耗时，聚合在 /metrics 里，这是后面做成本分析的数据来源。
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from app.foundation.settings import settings

logger = logging.getLogger(__name__)

Message = dict[str, str]


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    by_purpose: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_latency_ms": round(self.total_latency_ms / self.calls, 1) if self.calls else 0.0,
            "errors": self.errors,
            "by_purpose": dict(self.by_purpose),
        }


STATS = LLMStats()


class LLMError(RuntimeError):
    pass


# =============================================================================
# JSON 修复
# =============================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | list:
    """从模型输出里抠出 JSON。按代价从低到高试三种方式。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            text = m.group(1).strip()

    # 括号配平：从第一个 { 或 [ 开始扫，配平即截断。
    # 比贪婪正则可靠，能正确处理嵌套和字符串内的括号。
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"无法从模型输出解析 JSON：{text[:200]}")


# =============================================================================
# 真实客户端
# =============================================================================


class OpenAICompatClient:
    provider = "openai"

    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise LLMError("未配置 LLM_API_KEY")
        self.client = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout_s,
        )

    def _post(self, payload: dict) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                resp = self.client.post("/chat/completions", json=payload)
                if resp.status_code >= 500 or resp.status_code == 429:  # 429 限流，等一等就好
                    raise LLMError(f"上游 {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    # 4xx 是请求本身的问题，重试没意义
                    raise LLMError(f"请求错误 {resp.status_code}: {resp.text[:300]}")
                return resp
            # TransportError 覆盖超时、连接失败和 RemoteProtocolError（网关不回响应直接断开），
            # 只捕 NetworkError 会漏掉最后一种，评测跑到一半整条链路就断了
            except (httpx.TransportError, LLMError) as e:
                last = e
                if isinstance(e, LLMError) and "请求错误" in str(e):
                    raise
                if attempt < settings.llm_max_retries:
                    sleep = (2**attempt) * 0.5 + random.random() * 0.3
                    logger.warning("LLM 调用失败(%s)，%.1fs 后重试", type(e).__name__, sleep)
                    time.sleep(sleep)
        STATS.errors += 1
        raise LLMError(f"LLM 调用重试耗尽：{last}")

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        purpose: str = "chat",
        json_mode: bool = False,
    ) -> str:
        payload = {
            "model": model or settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        resp = self._post(payload)
        data = resp.json()
        _record(data.get("usage", {}), (time.perf_counter() - t0) * 1000, purpose)
        return data["choices"][0]["message"]["content"] or ""

    def chat_stream(
        self, messages: list[Message], *, model: str | None = None,
        temperature: float | None = None, max_tokens: int = 2048, purpose: str = "chat",
    ) -> Iterator[str]:
        payload = {
            "model": model or settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        t0 = time.perf_counter()
        with self.client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                body = line[6:]
                if body.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(body)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
        _record({}, (time.perf_counter() - t0) * 1000, purpose)


# =============================================================================
# Mock
# =============================================================================


class MockLLM:
    """确定性应答，用于离线跑通与单测。

    它不假装自己聪明 —— 回答里会显式标注 [MOCK]，避免评测时把 mock 结果
    误当成真实模型效果。需要结构化输出的地方返回合法但保守的 JSON。
    """

    provider = "mock"

    def chat(self, messages: list[Message], *, purpose: str = "chat", **kw) -> str:
        t0 = time.perf_counter()
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        out = self._respond(user, purpose)
        _record({"prompt_tokens": len(str(messages)) // 3, "completion_tokens": len(out) // 3},
                (time.perf_counter() - t0) * 1000, purpose)
        return out

    def chat_stream(self, messages: list[Message], *, purpose: str = "chat", **kw) -> Iterator[str]:
        text = self.chat(messages, purpose=purpose, **kw)
        for i in range(0, len(text), 12):
            yield text[i : i + 12]

    @staticmethod
    def _respond(user: str, purpose: str) -> str:
        if purpose == "agent":
            # Agent 循环里返回一个终止动作，避免 mock 模式下空转到 max_steps
            return json.dumps(
                {"thought": "[MOCK] 直接基于已有上下文作答", "action": "final_answer",
                 "action_input": {"answer": "[MOCK] 未配置真实 LLM，以下为占位回答。"
                                            "请在 .env 中设置 LLM_API_KEY 后重试。"}},
                ensure_ascii=False,
            )
        if purpose in ("extract_entity", "extract_relation"):
            return json.dumps({"entities": [], "relations": []}, ensure_ascii=False)
        if purpose == "summarize":
            return "[MOCK] 历史对话摘要占位。"
        if purpose == "plan":
            return json.dumps({"values": {}, "dimensions": {}, "rationale": {}}, ensure_ascii=False)
        head = user.strip().splitlines()[0][:60] if user.strip() else ""
        return (
            "[MOCK 回答] 当前未配置真实 LLM（LLM_API_KEY 为空），"
            "检索与图谱链路已正常执行，此处仅返回占位文本。\n"
            f"本轮问题：{head}\n"
            "配置 .env 中的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后可获得真实回答。"
        )


def _record(usage: dict, latency_ms: float, purpose: str) -> None:
    STATS.calls += 1
    STATS.prompt_tokens += int(usage.get("prompt_tokens", 0))
    STATS.completion_tokens += int(usage.get("completion_tokens", 0))
    STATS.total_latency_ms += latency_ms
    STATS.by_purpose[purpose] = STATS.by_purpose.get(purpose, 0) + 1


# =============================================================================
# 工厂 + JSON 封装
# =============================================================================

_CLIENT = None


def get_llm():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    choice = settings.llm_provider
    if choice in ("auto", "openai"):
        try:
            _CLIENT = OpenAICompatClient()
            logger.info("LLM: %s @ %s", settings.llm_model, settings.llm_base_url)
            return _CLIENT
        except LLMError as e:
            if choice == "openai":
                raise
            logger.warning("真实 LLM 不可用（%s），使用 MockLLM", e)
    _CLIENT = MockLLM()
    return _CLIENT


def chat_json(
    messages: list[Message], *, purpose: str = "chat", model: str | None = None,
    temperature: float | None = None, max_tokens: int = 2048, retry_on_parse_error: bool = True,
) -> dict | list:
    """要求模型返回 JSON，解析失败时带着错误信息让它自己修一次。"""
    llm = get_llm()
    kwargs = {"purpose": purpose, "model": model, "temperature": temperature,
              "max_tokens": max_tokens}
    if isinstance(llm, MockLLM):
        kwargs = {"purpose": purpose}
    else:
        kwargs["json_mode"] = True

    raw = llm.chat(messages, **kwargs)
    try:
        return extract_json(raw)
    except LLMError:
        if not retry_on_parse_error:
            raise
        logger.warning("JSON 解析失败，发起一次修复重试 (purpose=%s)", purpose)
        repair = [
            *messages,
            {"role": "assistant", "content": raw[:1500]},
            {"role": "user", "content": "上面的输出不是合法 JSON。请只输出合法 JSON，"
                                        "不要任何解释文字、不要 markdown 代码围栏。"},
        ]
        return extract_json(llm.chat(repair, **kwargs))


def llm_meta() -> dict:
    llm = get_llm()
    return {
        "provider": llm.provider,
        "model": settings.llm_model if llm.provider != "mock" else "mock",
        "base_url": settings.llm_base_url if llm.provider != "mock" else "-",
    }
