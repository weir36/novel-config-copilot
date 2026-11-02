"""ReAct Agent 循环。

为什么推荐/优化类问题要走 Agent 而不是一次性 RAG：

一次性 RAG 的流程是「检索 → 生成」，检索什么完全由用户的原始 query 决定。
但"给 B003 这本书做一套冲榜方案"这个问题，需要的信息是
  当前配置 → 可用的推荐和福利资源 → 每一项的依赖链 → 组合后是否违反 R001~R004 → 对曝光和收入的影响
这是一条**依赖前序结果才能决定下一步查什么**的链路，用户的原始 query 里没有这些信息。

所以这里做成 ReAct：LLM 每步只决定"现在调哪个工具"，看到结果后再决定下一步。
关键设计：

1. **校验是收尾的闸门。** 产出方案的意图每次 final_answer 都由系统从回答里抠出方案、
   合并书/作者的当前配置后校验：有 error 就把违规项退回让它改，最多 MAX_REPAIRS 轮。
   只靠提示词"给方案前必须校验"不够 —— 模型校验报错后照样会直接收尾。
   重改轮次用完仍不合规，pipeline 会在回答上明确标「未通过，请勿直接使用」。
2. **工具失败不中断。** 参数错了就把错误信息当作观察回灌，让它自己改参数重试。
3. **步数上限。** 超了就用已有观察强制收尾，不允许无限循环烧 token。
"""
from __future__ import annotations

import json
import logging
import time

from app.agent.tools import detect_scope, run_tool, tools_prompt
from app.foundation.settings import settings
from app.generation.llm_client import LLMError, chat_json, get_llm
from app.foundation.models import AgentStep, AgentTrace, Intent, ToolCall

logger = logging.getLogger(__name__)

AGENT_SYSTEM = """你是「网文作品资源配置助手」的推理内核，采用 思考-行动-观察 循环解决问题。

# 可用工具
{tools}

# 输出格式
每一步只输出一个 JSON 对象，不要输出任何其他文字：

{{"thought": "你现在的判断和下一步打算", "action": "工具名 或 final_answer",
  "action_input": {{...工具参数，或 final_answer 时为 {{"answer": "最终回答"}} ...}}}}

# 工作纪律

1. **不要凭记忆给取值。** 任何具体的取值范围、默认值、依赖关系，
   都必须先调 get_config_detail 确认。
2. **给配置方案前必须调 check_constraints。** 这是硬性要求。
   校验不通过就修改方案后重新校验，不要把有 error 的方案交出去。
   涉及某本具体的书或某个作者时，用 base_scope 参数（如 book:B003）把当前配置一起带进去校验 ——
   配错的情况大多是「新配项 × 没动的老项」的组合冲突，比如给买断书开完本奖，只校验新配项发现不了。
3. **做优化类任务前先调 get_current_config。** 不知道现状就说不清改了什么。
4. **量化代价。** 给出方案后调 estimate_impact，不要只说"会提升曝光"。
5. 工具返回错误时，读懂错误信息、修正参数后重试，不要重复同样的调用。
6. 信息够了就立刻 final_answer，不要为了多调几个工具而多调。

# 最终回答的格式要求
{answer_format}

# 本轮任务类型
{intent_hint}
"""

INTENT_HINTS = {
    Intent.RECOMMEND: "生成资源配置方案。顺序必须是：先看签约方式（R001 买断不发完本奖、"
                      "R003 保底作品分成不超过 50%）→ 再围绕目标选推荐和福利并补齐依赖链 "
                      "→ 开全勤奖要配日更字数且不低于 4000（R002）→ 最后完整校验。",
    Intent.OPTIMIZE: "在现有配置上调整。先查当前配置，只改必须改的，"
                     "并明确列出「保持不变」的项及原因。",
    Intent.EXPLAIN: "解释配置项或排查原因。以 get_config_detail 的精确定义为准，"
                    "用 search_knowledge 补充背景；问「为什么」时必须定位到具体配置项和规则编号。",
}

ANSWER_FORMATS = {
    Intent.RECOMMEND: (
        "## 方案目标\n## 配置清单（markdown 表格：配置项 | 取值 | 维度 | 理由，维度只能是 频道/作者/作品）\n"
        "## 约束检查（逐条说明依赖链、互斥、R001~R004 的校验结论）\n## 已知代价"
    ),
    Intent.OPTIMIZE: (
        "## 调整目标\n## 变更项（表格：配置项 | 当前值 | 建议值 | 理由）\n"
        "## 保持不变（列出考虑过但不动的项及原因）\n## 连带影响"
    ),
    Intent.EXPLAIN: "## 是什么（或：最可能的原因）\n## 取值\n## 约束（依赖链/互斥/规则编号）\n"
                    "## 影响（或：怎么解决）",
}

# 会产出配置方案的意图，这几类必须过校验
PLAN_PRODUCING = {Intent.RECOMMEND, Intent.OPTIMIZE}
# 校验不通过时退回重改的最多轮数。不占 max_steps：查资料用光步数后，仍然留有改方案的机会
MAX_REPAIRS = 2


class ConfigAgent:
    def __init__(self, max_steps: int | None = None) -> None:
        self.max_steps = max_steps or settings.agent_max_steps

    def run(
        self,
        question: str,
        intent: Intent,
        context: str = "",
        history: str = "",
        current_plan: str = "",
    ) -> AgentTrace:
        t0 = time.perf_counter()
        trace = AgentTrace()
        llm = get_llm()

        system = AGENT_SYSTEM.format(
            tools=tools_prompt(),
            intent_hint=INTENT_HINTS.get(intent, INTENT_HINTS[Intent.EXPLAIN]),
            answer_format=ANSWER_FORMATS.get(intent, ANSWER_FORMATS[Intent.EXPLAIN]),
        )
        user_parts = [f"【用户问题】\n{question}"]
        if context:
            user_parts.append(f"【预检索到的参考资料】\n{context[:3500]}")
        if history:
            user_parts.append(f"【历史对话】\n{history}")
        if current_plan:
            user_parts.append(f"【工作区当前方案】\n{current_plan}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        repairs = 0
        for step_no in range(1, self.max_steps + MAX_REPAIRS + 1):
            if step_no > self.max_steps + repairs:  # 没用上的重改轮次不能拿来多查资料
                trace.stopped_reason = "max_steps"
                trace.final_answer = self._forced_finish(messages, intent)
                break
            try:
                decision = chat_json(messages, purpose="agent", max_tokens=2500)
            except LLMError as e:
                logger.error("Agent 决策解析失败：%s", e)
                trace.stopped_reason = "error"
                trace.final_answer = _degrade_answer(trace, context)
                break

            if isinstance(decision, list):
                decision = decision[0] if decision else {}
            thought = str(decision.get("thought", ""))
            action = str(decision.get("action", "")).strip()
            action_input = decision.get("action_input") or {}
            if isinstance(action_input, str):
                try:
                    action_input = json.loads(action_input)
                except json.JSONDecodeError:
                    action_input = {"query": action_input}

            step = AgentStep(step=step_no, thought=thought)

            # ---- 收尾 ----
            if action in ("final_answer", "finish", "answer"):
                answer = str(action_input.get("answer") or action_input.get("content") or "")
                # 收尾闸门：方案有 error 就退回重改
                if intent in PLAN_PRODUCING and repairs < MAX_REPAIRS:
                    gate = self._force_validate(answer, question)
                    if gate is not None and not gate.data.get("passed", True):
                        repairs += 1
                        logger.info("方案未通过校验，退回重改（第 %d 轮）", repairs)
                        step.tool_call = ToolCall(name="check_constraints", arguments={"_forced": True})
                        step.tool_result = gate
                        trace.steps.append(step)
                        messages.append({"role": "assistant", "content": json.dumps(
                            decision, ensure_ascii=False)})
                        messages.append({"role": "user", "content":
                            f"【系统校验】你给出的方案未通过规则校验：\n{gate.content}\n\n"
                            f"请修改方案消除所有 error，然后重新给出完整的 final_answer。"})
                        continue
                trace.steps.append(step)
                trace.final_answer = answer
                trace.stopped_reason = "final"
                break

            # ---- 调工具 ----
            if action not in ("", "none"):
                step.tool_call = ToolCall(name=action, arguments=action_input)
                result = run_tool(action, action_input)
                step.tool_result = result
                trace.steps.append(step)

                messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
                messages.append({
                    "role": "user",
                    "content": f"【观察 - {action}】\n{result.content[:3500]}\n\n"
                               f"基于这个结果继续。信息足够就给出 final_answer。",
                })
                continue

            # ---- 模型没给出有效动作 ----
            trace.steps.append(step)
            messages.append({"role": "user", "content":
                "你的输出没有包含有效的 action。请严格按 JSON 格式输出，"
                "action 必须是工具名或 final_answer。"})
        else:
            trace.stopped_reason = "max_steps"
            trace.final_answer = self._forced_finish(messages, intent)

        trace.total_latency_ms = (time.perf_counter() - t0) * 1000
        return trace

    # ---------------- 收尾策略 ----------------
    def _force_validate(self, answer: str, question: str):
        """从自然语言回答里抠出配置方案，合并问题里那本书 / 那个作者的当前配置后校验。"""
        from app.validation.plan_extract import extract_plan  # noqa: PLC0415

        plan = extract_plan(answer)
        if not plan.values:
            return None
        args = {"values": plan.values, "dimensions": plan.dimensions}
        if scope := detect_scope(question):
            args["base_scope"] = scope
        return run_tool("check_constraints", args)

    def _forced_finish(self, messages: list[dict], intent: Intent) -> str:
        """步数用尽时的强制收尾：用已有观察生成回答，并明确告知用户推理被截断。"""
        messages.append({
            "role": "user",
            "content": "推理步数已达上限。请基于目前已有的观察结果，直接输出最终回答，"
                       "不要再调用工具。若信息不足以给出完整方案，明确说明还缺什么。"
                       "只输出回答正文，不要 JSON。",
        })
        try:
            answer = get_llm().chat(messages, purpose="agent_finish", max_tokens=2500)
        except LLMError:
            return "推理步数已达上限且收尾失败，请重新提问或缩小问题范围。"
        return answer + "\n\n> ⚠️ 本轮推理达到步数上限，以上结论基于已获取的部分信息，建议复核。"


def _degrade_answer(trace: AgentTrace, context: str) -> str:
    """Agent 完全跑不动时的降级：把检索到的资料原样给用户，好过什么都不给。"""
    observed = "\n\n".join(
        s.tool_result.content[:800] for s in trace.steps if s.tool_result and s.tool_result.ok
    )
    body = observed or context[:2000]
    return (
        "推理过程出现异常，以下是本轮检索到的相关资料，供你参考：\n\n"
        + (body or "（未检索到资料）")
    )


_AGENT: ConfigAgent | None = None


def get_agent() -> ConfigAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = ConfigAgent()
    return _AGENT
