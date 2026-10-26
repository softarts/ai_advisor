#!/usr/bin/env python3
"""Evaluation runner for the real advisor agent.

Runs a fixed case set against the live agent (real LLM + real tools + real
answer checklist), combining deterministic assertions with a separate semantic
judge for explicitly configured rubrics. Each case runs in its own thread.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.agent import Advisor
from advisor.llm_settings import MAX_COMPLETION_TOKENS
from advisor.output_review import CheckContext
from advisor.tools import create_tools
from eval.report import CaseResult, evaluate_case, summarize
from eval.semantic_judge import SemanticJudgeError, review_case, select_tool_facts

REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
CASES_DIR = Path(__file__).resolve().parent / "cases"
NOTE_MARKER = "⚠️"


def load_env_from_project_root() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def ensure_llm_settings() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL before running the evaluation."
        )


def preflight_llm() -> str:
    """Send one minimal request so misconfiguration fails fast with a clear reason.

    Returns a human-readable status line. Raises RuntimeError when the endpoint
    or model is unusable, because every case would otherwise fail for the same
    reason and the real cause would be hidden behind content assertions.
    """
    import httpx

    base_url = os.environ["LLM_BASE_URL"].rstrip("/")
    model = os.environ["LLM_MODEL"]
    url = base_url + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "reply with OK"}],
        "max_tokens": MAX_COMPLETION_TOKENS,
    }
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=120)
    except httpx.RequestError as exc:
        raise RuntimeError(f"无法连接 LLM 端点 {url}：{exc}") from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"端点 {url} 返回 404：通常表示 LLM_MODEL 的值 {model!r} 不被该服务端接受。"
            "请核对模型名（NVIDIA NIM 可用模型见 /v1/models 列表）。"
        )
    if response.status_code == 410:
        raise RuntimeError(
            f"端点 {url} 返回 410：模型 {model!r} 已下线，请改用当前可用的模型。"
        )
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"端点 {url} 返回 {response.status_code}：LLM_API_KEY 无效或无该模型权限。"
        )
    if response.status_code >= 400:
        detail = response.text[:300]
        hint = ""
        if response.status_code == 500 and "Failed to generate completions" in detail:
            hint = (
                " 该错误常由 max_tokens 过小触发：推理模型的 reasoning 阶段会先消耗预算，"
                "预算过小会让服务端在生成阶段失败。"
            )
        raise RuntimeError(
            f"端点 {url} 返回 {response.status_code}：{detail}{hint}"
        )
    return f"端点与模型可用：{model} @ {url}"


def preflight_tool_support(attempts: int = 3) -> str:
    """Check that the configured model can be called with a tool schema.

    Some models answer plain questions but fail the moment a tool schema is
    attached, and the gateway reports that as an opaque 500. The agent needs
    tool calling, so this is checked before any case runs.

    The gateway occasionally returns 500 under load, so a transient failure is
    retried before it is reported as "this model cannot use tools".
    """
    import time

    import httpx

    from advisor.tools import create_tools
    from pa_analytics import Store

    base_url = os.environ["LLM_BASE_URL"].rstrip("/")
    model = os.environ["LLM_MODEL"]
    url = base_url + "/chat/completions"
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    schema = [t.api_schema() for t in create_tools(Store("data/pa.db", "data/raw"))]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "查 2025-03 的月度明细"}],
        "tools": schema,
        "max_tokens": MAX_COMPLETION_TOKENS,
    }
    last_status = 0
    last_body = ""
    for attempt in range(attempts):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=120)
        except httpx.RequestError as exc:
            last_status, last_body = 0, str(exc)
        else:
            if response.status_code < 400:
                return f"工具调用可用：{model}"
            last_status, last_body = response.status_code, response.text[:200]
        if attempt < attempts - 1:
            time.sleep(1.0)
    raise RuntimeError(
        f"模型 {model!r} 无法处理带工具 schema 的请求（HTTP {last_status}）：{last_body}。"
        "本项目依赖工具调用，请改用支持 tool calling 的模型"
        "（例如 nvidia/nemotron-3.5-lightning-30b-a3b）。"
    )


def load_cases(case_filter: list[str] | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "legacy_not_loaded":
            continue
        intent = payload.get("intent", path.stem)
        for case in payload.get("cases", []):
            case = {**case, "intent": intent, "case_file": path.name}
            if case_filter and case["id"] not in case_filter:
                continue
            cases.append(case)
    return cases


class EventCollector(logging.Handler):
    """Captures the agent's structured JSON log events for assertions."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.events.append(json.loads(record.getMessage()))
        except (json.JSONDecodeError, TypeError):
            pass


def _run_logger() -> tuple[logging.Logger, EventCollector]:
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    collector = EventCollector()
    logger.addHandler(collector)
    return logger, collector


def _events_for(collector: EventCollector, trace_id: str) -> list[dict[str, Any]]:
    return [event for event in collector.events if event.get("trace_id") == trace_id]


def _llm_failure_detail(collector: EventCollector) -> str:
    """Summarize the last failed LLM call: request shape plus response body.

    A 5xx from the model gateway is usually caused by the request shape
    (model, token cap, tool schema) rather than by the question, so the
    summary names the request parameters and the server's own message.
    """
    request: dict[str, Any] = {}
    for event in collector.events:
        if event.get("event") == "llm_http_request":
            request = event
    summary = ""
    if request:
        payload = request.get("payload") or {}
        tools = payload.get("tools") or []
        summary = (
            f"请求模型={payload.get('model')!r}"
            f" max_tokens={payload.get('max_tokens')!r}"
            f" tools={len(tools)}"
            f" messages={len(payload.get('messages') or [])}"
            f" intent={request.get('intent')!r}"
        )
    for event in reversed(collector.events):
        if event.get("event") != "llm_http_response":
            continue
        status = event.get("status")
        if isinstance(status, int) and status >= 400:
            body = str(event.get("body") or "").strip()
            return f"HTTP {status} {body[:400]}" + (f"\n    {summary}" if summary else "")
    return summary


def diagnose_llm() -> list[str]:
    """Probe which request feature the endpoint rejects.

    Sends the same question with progressively richer payloads so a 5xx can be
    attributed to a specific feature (tool schema, token cap, long prompt)
    rather than guessed at.
    """
    import httpx

    from advisor.llm_settings import MAX_COMPLETION_TOKENS
    from advisor.router import load_compose_template
    from advisor.tools import create_tools
    from pa_analytics import Store

    store = Store("data/pa.db", "data/raw")
    tools = create_tools(store)
    schema = [t.api_schema() for t in tools]
    compose = load_compose_template("data_query")
    question = "为什么 2025-03 收益这么高？"

    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    model = os.environ["LLM_MODEL"]

    probes = [
        ("A 纯对话，无 tools", {"model": model, "messages": [
            {"role": "system", "content": compose},
            {"role": "user", "content": question},
        ]}),
        ("B 纯对话 + max_tokens", {"model": model, "messages": [
            {"role": "system", "content": compose},
            {"role": "user", "content": question},
        ], "max_tokens": MAX_COMPLETION_TOKENS}),
        ("C 短 system + 11 tools", {"model": model, "messages": [
            {"role": "user", "content": question},
        ], "tools": schema, "max_tokens": MAX_COMPLETION_TOKENS}),
        ("D 完整 system + 11 tools（等同生产）", {"model": model, "messages": [
            {"role": "system", "content": compose},
            {"role": "user", "content": question},
        ], "tools": schema, "max_tokens": MAX_COMPLETION_TOKENS}),
        ("E 完整 system + 11 tools，不设 max_tokens", {"model": model, "messages": [
            {"role": "system", "content": compose},
            {"role": "user", "content": question},
        ], "tools": schema}),
    ]

    lines = []
    for label, payload in probes:
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=120)
            body = response.text[:220].replace("\n", " ")
            lines.append(f"{response.status_code}  {label}\n      {body}")
        except httpx.RequestError as exc:
            lines.append(f"EXC  {label}\n      {exc}")
    return lines


def _extract_notes(answer: str) -> list[str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip().startswith(NOTE_MARKER)]
    return lines


def run_case(case: dict[str, Any], advisor: Advisor, collector: EventCollector) -> CaseResult:
    trace_id = uuid.uuid4().hex
    thread_id = f"eval-{uuid.uuid4().hex}"
    answer = advisor.ask(case["question"], thread_id, trace_id)
    events = _events_for(collector, trace_id)

    tool_events = [event for event in events if event.get("event") == "tool_call"]
    actual_tools = [event.get("tool") for event in tool_events]
    tool_results = [
        {"tool": event.get("tool", "tool"), "content": json.dumps(event.get("result"), ensure_ascii=False)}
        for event in events if event.get("event") == "tool_result"
    ]
    review_events = [event for event in events if event.get("event") == "output_review_summary"]
    review = review_events[-1] if review_events else {}
    router_events = [event for event in events if event.get("event") == "router_classified"]
    actual_intent = router_events[-1].get("intent", "") if router_events else ""
    router_fallbacks = [event for event in events if event.get("event") == "router_fallback_unsupported"]
    if router_fallbacks and not actual_intent:
        result_failure = f"意图分类失败并回退 unsupported：{router_fallbacks[-1].get('error', '未知原因')}"
    else:
        result_failure = ""

    result = CaseResult(
        case_id=case["id"],
        name=case.get("name", case["id"]),
        intent=case.get("intent", ""),
        question=case["question"],
        answer=answer,
        actual_tools=actual_tools,
        tool_results=tool_results,
        review=review,
        notes=_extract_notes(answer),
        actual_intent=actual_intent,
    )
    rubric = case.get("semantic_rubric") or []
    if rubric:
        try:
            semantic_facts = select_tool_facts(
                tool_results, case.get("semantic_fact_paths") or []
            )
            result.semantic_review = review_case(
                case_id=case["id"],
                question=case["question"],
                answer=answer,
                tool_results=semantic_facts,
                rubric=rubric,
            )
        except SemanticJudgeError as exc:
            result.semantic_review = {"status": "unavailable", "error": str(exc)}
    if result_failure:
        result.failures.append(result_failure)
    return evaluate_case(case, result)


def print_case_report(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n[{status}] {result.case_id} {result.name}")
    print(f"  意图: {result.intent}" + (
        f"（实际分类：{result.actual_intent}）" if result.actual_intent and result.actual_intent != result.intent
        else ""
    ))
    print(f"  工具: {result.actual_tools or '无'}")
    review = result.review or {}
    if review:
        print(f"  检查清单: 通过={review.get('passed')} 运行={review.get('checks_run')} "
              f"失败={len(review.get('failed_checks') or [])} 不可用={len(review.get('unavailable_checks') or [])}")
    else:
        print("  检查清单: 未执行（回答未经过 output_review 节点）")
    semantic = result.semantic_review
    if semantic:
        semantic_items = semantic.get("results") or []
        print(
            f"  语义评审: 状态={semantic.get('status')} "
            f"通过={sum(item.get('decision') == 'pass' for item in semantic_items)} "
            f"失败={sum(item.get('decision') == 'fail' for item in semantic_items)} "
            f"不确定={sum(item.get('decision') == 'uncertain' for item in semantic_items)}"
        )
        if semantic.get("model"):
            print(
                f"    Judge={semantic['model']} "
                f"prompt_version={semantic.get('prompt_version', 'unknown')}"
            )
        if semantic.get("error"):
            print(f"    不可用原因：{semantic['error']}")
        for item in semantic_items:
            if item.get("decision") != "pass":
                evidence = item.get("evidence") or "（未找到直接证据）"
                print(
                    f"    {item.get('id')} [{item.get('decision')}]: "
                    f"{item.get('reason', '')}；证据：{evidence}"
                )
    for failure in result.failures:
        print(f"  ✗ {failure}")


def print_case_answer(result: CaseResult) -> None:
    print(f"  原始问题: {result.question}")
    print("  --- 回答 ---")
    print(result.answer)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation case set against the live agent.")
    parser.add_argument("--case", action="append", help="Run only the given case id (repeatable)")
    parser.add_argument("--db", default="data/pa.db")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--json-out", help="Write the detailed results as JSON to this path")
    parser.add_argument("--show-answer", action="store_true", help="Print the full answer of every case")
    parser.add_argument("--diagnose", action="store_true",
                        help="Probe which request feature the endpoint rejects, then exit")
    args = parser.parse_args(argv)

    load_env_from_project_root()
    ensure_llm_settings()
    if args.diagnose:
        print("端点诊断（逐项加码请求，定位 5xx 来自哪个特性）：")
        for line in diagnose_llm():
            print("  " + line)
        return 0
    print(preflight_llm())
    print(preflight_tool_support())

    from pa_analytics import Store

    cases = load_cases(args.case)
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 1

    store = Store(args.db, args.raw_dir)
    tools = create_tools(store)
    logger, collector = _run_logger()
    advisor = Advisor(
        tools,
        logger,
        timezone_name=os.environ.get("APP_TIMEZONE", "Asia/Singapore"),
        data_as_of=lambda: store.current().period_end,
    )

    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"\n=== [{index}/{len(cases)}] {case['id']} {case.get('name', '')} ===")
        try:
            result = run_case(case, advisor, collector)
        except Exception as exc:
            detail = _llm_failure_detail(collector)
            result = CaseResult(
                case_id=case["id"],
                name=case.get("name", case["id"]),
                intent=case.get("intent", ""),
                question=case["question"],
                answer="",
                semantic_review=(
                    {"status": "unavailable", "error": "案例运行异常，未生成可评审回答"}
                    if case.get("semantic_rubric") else {}
                ),
                failures=[f"运行异常：{exc}" + (f"\n    服务端响应：{detail}" if detail else "")],
            )
        results.append(result)
        print_case_report(result)
        if args.show_answer:
            print_case_answer(result)

    summary = summarize(results)
    print("\n" + "=" * 60)
    print(f"总用例: {summary['total']}  通过: {summary['passed']}  通过率: {summary['pass_rate']}%")
    for intent, bucket in sorted(summary["by_intent"].items()):
        rate = round(bucket["passed"] / bucket["total"] * 100, 1) if bucket["total"] else 0.0
        print(f"  {intent}: {bucket['passed']}/{bucket['total']} ({rate}%)")
    semantic_summary = summary["semantic_review"]
    routing_summary = summary["routing"]
    print(
        "路由分类: "
        f"正确={routing_summary['correct']} 错误={routing_summary['incorrect']} "
        f"未记录={routing_summary['unrecorded']}"
    )
    if semantic_summary["cases"]:
        print(
            "语义评审汇总: "
            f"通过={semantic_summary['pass']} 失败={semantic_summary['fail']} "
            f"不确定={semantic_summary['uncertain']} 不可用={semantic_summary['unavailable']}"
        )
    if summary["failure_reasons"]:
        print("失败原因分布:")
        for reason, count in summary["failure_reasons"].items():
            print(f"  {count:>3} x {reason}")
    if summary["derived_numbers"] or summary["unexplained_numbers"]:
        print(f"数字溯源: 模型自行推导 {summary['derived_numbers']} 个，"
              f"无法解释来源 {summary['unexplained_numbers']} 个")

    if args.json_out:
        payload = {
            "summary": summary,
            "results": [
                {
                    "id": r.case_id,
                    "name": r.name,
                    "intent": r.intent,
                    "question": r.question,
                    "answer": r.answer,
                    "actual_tools": r.actual_tools,
                    "review": r.review,
                    "semantic_review": r.semantic_review,
                    "notes": r.notes,
                    "failures": r.failures,
                    "passed": r.passed,
                }
                for r in results
            ],
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n详细结果已写入 {args.json_out}")

    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
