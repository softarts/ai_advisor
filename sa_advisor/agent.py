"""StrandsAdvisor — Strands-SDK portfolio advisor.

Self-contained: imports only from common/, sa_advisor/, and the strands SDK.
No dependency on advisor/.

Public interface matches advisor.Advisor so the CLI and tests can swap
implementations without changes.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from strands import Agent
from strands.handlers import null_callback_handler

from common.prompts import (
    INTENT_PROFILES,
    extract_fixed_reply,
    load_compose_template,
    load_output_review_config,
)
from common.output_review import CHECK_PLUGINS, CheckContext
from common.checklist_config import GLOBAL_CHECKS, GLOBAL_CHECKLIST, INTENT_CHECKLISTS
from common.tools import Tool

from .model_provider import get_model
from .router import classify_intent
from .tools import wrap_tools


def _extract_text(result) -> str:
    """Pull the text string out of an AgentResult."""
    return str(result).strip()


def _extract_tool_results(messages: list) -> list[dict[str, str]]:
    """Walk message history and collect every toolResult content block."""
    # Build id→name map from toolUse blocks first
    id_to_name: dict[str, str] = {}
    for msg in messages:
        for block in msg.get("content", []):
            tu = block.get("toolUse")
            if tu:
                id_to_name[tu["toolUseId"]] = tu.get("name", "tool")

    results: list[dict[str, str]] = []
    for msg in messages:
        for block in msg.get("content", []):
            tr = block.get("toolResult")
            if not tr:
                continue
            tool_name = id_to_name.get(tr.get("toolUseId", ""), "tool")
            content_str = ""
            for cb in tr.get("content", []):
                if "text" in cb:
                    content_str = cb["text"]
                    break
                if "json" in cb:
                    content_str = json.dumps(cb["json"], ensure_ascii=False)
                    break
            results.append({"tool": tool_name, "content": content_str})
    return results


class StrandsAdvisor:
    """Portfolio advisor built on the Strands Agents SDK.

    Maintains one Strands Agent per conversation thread so multi-turn history
    is preserved natively via agent.messages.
    """

    def __init__(
        self,
        tools: list[Tool],
        logger: logging.Logger,
        user_id: str = "default",
        timezone_name: str | None = None,
        data_as_of=None,
    ):
        self._advisor_tools = {t.name: t for t in tools}
        self._strands_tools = {t.tool_name: t for t in wrap_tools(tools)}
        self.logger = logger
        self.user_id = user_id
        self.timezone_name = timezone_name or os.environ.get("APP_TIMEZONE", "Asia/Singapore")
        self.data_as_of = data_as_of
        self._memory: dict[str, str] = {}
        # One Agent per thread_id; keyed so history persists across ask() calls
        self._agents: dict[str, Agent] = {}

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        self._memory[key] = value

    def recall(self) -> dict:
        return dict(self._memory)

    # ------------------------------------------------------------------
    # Core ask
    # ------------------------------------------------------------------

    def ask(self, text: str, thread_id: str, trace_id: str | None = None) -> str:
        """Process a user question and return the answer string."""
        intent = classify_intent(text, self.logger, trace_id)

        if intent == "unsupported":
            template = load_compose_template("unsupported")
            reply = extract_fixed_reply(template)
            if not reply:
                reply = "很抱歉，您的问题超出了本系统的服务范围。"
            return reply

        system_prompt = self._build_system_prompt(intent, repair_request="")
        agent = self._get_or_create_agent(thread_id, intent, system_prompt)
        result = agent(text)
        candidate = _extract_text(result)

        candidate = self._run_output_review(
            agent=agent,
            intent=intent,
            question=text,
            candidate=candidate,
            trace_id=trace_id,
            repair_count=0,
        )
        return candidate

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def _get_or_create_agent(self, thread_id: str, intent: str, system_prompt: str) -> Agent:
        """Return the existing agent for thread_id, updating its system_prompt."""
        profile = INTENT_PROFILES.get(intent, INTENT_PROFILES["data_query"])
        allowed_tool_names = set(profile["allowed_tools"])
        filtered_tools = [
            t for name, t in self._strands_tools.items()
            if name in allowed_tool_names
        ]

        if thread_id not in self._agents:
            self._agents[thread_id] = Agent(
                model=get_model(),
                tools=filtered_tools,
                system_prompt=system_prompt,
                callback_handler=null_callback_handler,
            )
        else:
            agent = self._agents[thread_id]
            # Update system prompt in-place to preserve message history
            try:
                from strands.agent.agent import split_system_prompt
                _, content = split_system_prompt(system_prompt)
                agent._system_prompt_content = content
            except Exception:
                pass
            # Update tool registry for this intent
            try:
                from strands.tools.registry import ToolRegistry
                agent._tools = ToolRegistry(
                    tools=filtered_tools,
                    load_tools_from_directory=False,
                )
            except Exception:
                pass

        return self._agents[thread_id]

    def _build_system_prompt(self, intent: str, repair_request: str = "") -> str:
        """Compose the system prompt for the given intent."""
        now = datetime.now(ZoneInfo(self.timezone_name))
        as_of = self.data_as_of() if callable(self.data_as_of) else self.data_as_of

        time_context = (
            f"\n本轮当前日期时间：{now.isoformat(timespec='seconds')}（时区 {self.timezone_name}）。"
            f"账户报告数据截止日：{as_of or '未知'}。当前日期与数据截止日不同，不能假设报告已更新到今天。"
        )

        compose = load_compose_template(intent)
        system = compose + time_context if compose else (
            "你是投资组合数据助手。账户相关问题必须用工具取得数据，不能自行计算工具可计算的数字。"
            + time_context
        )

        if self._memory:
            memory_text = "\n".join(f"{k}: {v}" for k, v in self._memory.items())
            system += "\n用户长期记忆：\n" + memory_text

        if repair_request:
            system += "\n\n" + repair_request

        return system

    # ------------------------------------------------------------------
    # Output review
    # ------------------------------------------------------------------

    def _run_output_review(
        self,
        agent: Agent,
        intent: str,
        question: str,
        candidate: str,
        trace_id: str | None,
        repair_count: int,
    ) -> str:
        if not candidate.strip():
            return candidate

        tool_results = _extract_tool_results(agent.messages)
        review_config = load_output_review_config(intent)
        context = CheckContext(
            intent=intent,
            question=question,
            candidate=candidate,
            tool_results=tool_results,
            logger=self.logger,
            trace_id=trace_id,
            review_config=review_config,
        )

        failed_checks: list[tuple[str, str, str]] = []
        unavailable_checks: list[tuple[str, str, str]] = []
        checks_run: list[str] = []

        if intent not in INTENT_CHECKLISTS:
            self.logger.error(json.dumps({
                "event": "output_review_configuration_error",
                "trace_id": trace_id,
                "intent": intent,
                "error": "intent has no configured checklist",
            }, ensure_ascii=False))
            unavailable_checks.append((
                "checklist", "答案检查配置缺失，数据准确性无法确认。",
                "intent has no configured checklist",
            ))

        for check_id in dict.fromkeys(GLOBAL_CHECKLIST + INTENT_CHECKLISTS.get(intent, [])):
            definition = GLOBAL_CHECKS.get(check_id)
            checker = CHECK_PLUGINS.get(check_id)
            if definition is None or checker is None:
                error = f"unknown or unregistered check: {check_id}"
                self.logger.error(json.dumps({
                    "event": "output_review_configuration_error",
                    "trace_id": trace_id,
                    "intent": intent,
                    "check_id": check_id,
                    "error": error,
                }, ensure_ascii=False))
                note = definition["unavailable_note"] if definition else error
                unavailable_checks.append((check_id, note, error))
                continue
            if not definition["enabled"]:
                continue
            try:
                outcome = checker(context)
                if outcome.candidate is not None:
                    context.candidate = outcome.candidate
                checks_run.append(check_id)
                if not outcome.passed:
                    failed_checks.append((check_id, outcome.note or definition["failure_note"], outcome.reason))
                self.logger.info(json.dumps({
                    "event": "output_review_check",
                    "trace_id": trace_id,
                    "intent": intent,
                    "check_id": check_id,
                    "passed": outcome.passed,
                    "reason": outcome.reason,
                }, ensure_ascii=False))
            except Exception as exc:
                self.logger.error(json.dumps({
                    "event": "output_review_check_error",
                    "trace_id": trace_id,
                    "intent": intent,
                    "check_id": check_id,
                    "error": str(exc),
                }, ensure_ascii=False))
                unavailable_checks.append((check_id, definition["unavailable_note"], str(exc)))

        notes = [note for _, note, _ in failed_checks + unavailable_checks if note]
        numeric_failures = [
            reason for check_id, _, reason in failed_checks
            if check_id == "numeric_provenance"
        ]

        # Attempt one repair if numeric provenance failed on the first pass
        if repair_count < 1 and numeric_failures:
            repair_instructions = (
                "数字溯源复核发现以下金额或比率没有本轮工具结果的直接依据："
                + "；".join(numeric_failures)
                + "。删除这些无依据数字及依赖它们的结论；只保留工具结果直接支持的金额和比率，"
                "保持来源精度，不要自行计算、汇总、换算、取整或近似。"
            )
            repair_request = (
                "请根据以下检查结果修订上一版完整回答："
                + repair_instructions
                + "不得调用工具，不要提及本次复核或输出检查说明。"
            )
            self.logger.info(json.dumps({
                "event": "output_review_repair_requested",
                "trace_id": trace_id,
                "intent": intent,
                "retry_count": 1,
                "failed_checks": ["numeric_provenance"],
                "reasons": numeric_failures,
            }, ensure_ascii=False))

            repair_agent = Agent(
                model=get_model(),
                messages=list(agent.messages),
                tools=[],
                system_prompt=self._build_system_prompt(intent, repair_request=repair_request),
                callback_handler=null_callback_handler,
            )
            repair_result = repair_agent(repair_request)
            repaired_candidate = _extract_text(repair_result)
            return self._run_output_review(
                agent=repair_agent,
                intent=intent,
                question=question,
                candidate=repaired_candidate,
                trace_id=trace_id,
                repair_count=1,
            )

        if notes:
            context.candidate += "\n\n" + "\n".join(
                dict.fromkeys(f"⚠️ {note}" for note in notes)
            )

        self.logger.info(json.dumps({
            "event": "output_review_summary",
            "trace_id": trace_id,
            "intent": intent,
            "checks_run": checks_run,
            "failed_checks": [
                {"check_id": cid, "reason": reason}
                for cid, _, reason in failed_checks
            ],
            "unavailable_checks": [
                {"check_id": cid, "reason": reason}
                for cid, _, reason in unavailable_checks
            ],
            "notes": notes,
            "annotated": context.candidate != candidate,
            "passed": not failed_checks and not unavailable_checks,
        }, ensure_ascii=False))

        return context.candidate
