#!/usr/bin/env python3
"""Evaluation runner for the SA Advisor (Strands SDK).

Mirrors eval/runner.py but targets StrandsAdvisor instead of Advisor.
Tool calls and results are extracted from agent.messages (Strands message
history) rather than from log events, because the Strands SDK does not emit
tool_call/tool_result log events the way the LangGraph advisor does.
All other event types (output_review_summary, router_classified,
router_fallback_unsupported) are logged as JSON strings by the agent and
are collected via EventCollector exactly as in eval/runner.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bootstrap_strands() -> None:
    """Make the real strands-agents SDK importable as 'strands'.

    Temporarily removes the workspace from sys.path so the installed SDK is
    found instead of our local strands/ directory, loads it, then restores
    sys.path and replants the SDK under the 'strands' name so that all
    subsequent 'from strands import ...' calls resolve correctly.
    """
    existing = sys.modules.get("strands")
    if existing is not None and "site-packages" in (getattr(existing, "__file__", "") or ""):
        return

    def _is_workspace(p: str) -> bool:
        if not p:
            return True
        try:
            return os.path.realpath(p) == os.path.realpath(_WORKSPACE)
        except Exception:
            return False

    _orig_path = sys.path[:]
    sys.path = [p for p in sys.path if not _is_workspace(p)]

    _shadowed = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "strands" or k.startswith("strands.")}
    try:
        import strands as _sdk
        sys.modules["_strands_sdk"] = _sdk
    except Exception:
        sys.path = _orig_path
        for k, v in _shadowed.items():
            sys.modules[k] = v
        return
    finally:
        sys.path = _orig_path


_bootstrap_strands()

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_advisor.agent import StrandsAdvisor
from common.tools import create_tools
from eval.report import CaseResult, evaluate_case, summarize
from eval.semantic_judge import SemanticJudgeError, review_case, select_tool_facts

REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
# Cases are shared with the LangGraph advisor runner
CASES_DIR = Path(__file__).resolve().parents[2] / "eval" / "cases"
NOTE_MARKER = "\u26a0\ufe0f"


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
    """Send one minimal request so misconfiguration fails fast with a clear reason."""
    import httpx

    base_url = os.environ["LLM_BASE_URL"].rstrip("/")
    model = os.environ["LLM_MODEL"]
    url = base_url + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "reply with OK"}],
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=120)
    except httpx.RequestError as exc:
        raise RuntimeError(f"\u65e0\u6cd5\u8fde\u63a5 LLM \u7aef\u70b9 {url}\uff1a{exc}") from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"\u7aef\u70b9 {url} \u8fd4\u56de 404\uff1a\u901a\u5e38\u8868\u793a LLM_MODEL \u7684\u503c {model!r} \u4e0d\u88ab\u8be5\u670d\u52a1\u7aef\u63a5\u53d7\u3002"
            "\u8bf7\u6838\u5bf9\u6a21\u578b\u540d\uff08NVIDIA NIM \u53ef\u7528\u6a21\u578b\u89c1 /v1/models \u5217\u8868\uff09\u3002"
        )
    if response.status_code == 410:
        raise RuntimeError(
            f"\u7aef\u70b9 {url} \u8fd4\u56de 410\uff1a\u6a21\u578b {model!r} \u5df2\u4e0b\u7ebf\uff0c\u8bf7\u6539\u7528\u5f53\u524d\u53ef\u7528\u7684\u6a21\u578b\u3002"
        )
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"\u7aef\u70b9 {url} \u8fd4\u56de {response.status_code}\uff1aLLM_API_KEY \u65e0\u6548\u6216\u65e0\u8be5\u6a21\u578b\u6743\u9650\u3002"
        )
    if response.status_code >= 400:
        detail = response.text[:300]
        raise RuntimeError(
            f"\u7aef\u70b9 {url} \u8fd4\u56de {response.status_code}\uff1a{detail}"
        )
    return f"\u7aef\u70b9\u4e0e\u6a21\u578b\u53ef\u7528\uff1a{model} @ {url}"


def preflight_tool_support(attempts: int = 3) -> str:
    """Check that the configured model can be called with a tool schema."""
    import time
    import httpx
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
        "messages": [{"role": "user", "content": "\u67e5 2025-03 \u7684\u6708\u5ea6\u660e\u7ec6"}],
        "tools": schema,
        "max_tokens": 2048,
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
                return f"\u5de5\u5177\u8c03\u7528\u53ef\u7528\uff1a{model}"
            last_status, last_body = response.status_code, response.text[:200]
        if attempt < attempts - 1:
            time.sleep(1.0)
    raise RuntimeError(
        f"\u6a21\u578b {model!r} \u65e0\u6cd5\u5904\u7406\u5e26\u5de5\u5177 schema \u7684\u8bf7\u6c42\uff08HTTP {last_status}\uff09\uff1a{last_body}\u3002"
        "\u672c\u9879\u76ee\u4f9d\u8d56\u5de5\u5177\u8c03\u7528\uff0c\u8bf7\u6539\u7528\u652f\u6301 tool calling \u7684\u6a21\u578b\u3002"
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
    """Captures the agent's structured JSON log events for assertions.

    Handles both JSON-string messages (from agent.py) and dict messages
    (from router.py which calls logger.info({...}) directly).
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if isinstance(msg, dict):
            self.events.append(msg)
            return
        try:
            self.events.append(json.loads(msg))
        except (json.JSONDecodeError, TypeError):
            pass


def _run_logger() -> tuple[logging.Logger, EventCollector]:
    logger = logging.getLogger("sa_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    collector = EventCollector()
    logger.addHandler(collector)
    return logger, collector


def _events_for(collector: EventCollector, trace_id: str) -> list[dict[str, Any]]:
    return [event for event in collector.events if event.get("trace_id") == trace_id]


def _extract_tools_from_messages(messages: list) -> tuple[list[str], list[dict[str, str]]]:
    """Extract tool names and results from Strands agent.messages.

    Returns (actual_tools, tool_results) where tool_results matches the
    format expected by eval.report and eval.semantic_judge:
    [{"tool": name, "content": json_string}, ...]
    """
    id_to_name: dict[str, str] = {}
    for msg in messages:
        for block in msg.get("content", []):
            tu = block.get("toolUse")
            if tu:
                id_to_name[tu["toolUseId"]] = tu.get("name", "tool")

    actual_tools: list[str] = []
    tool_results: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for msg in messages:
        for block in msg.get("content", []):
            # Collect tool call order from toolUse blocks
            tu = block.get("toolUse")
            if tu and tu["toolUseId"] not in seen_ids:
                seen_ids.add(tu["toolUseId"])
                actual_tools.append(tu.get("name", "tool"))
            # Collect tool results
            tr = block.get("toolResult")
            if tr:
                tool_name = id_to_name.get(tr.get("toolUseId", ""), "tool")
                content_str = ""
                for cb in tr.get("content", []):
                    if "text" in cb:
                        content_str = cb["text"]
                        break
                    if "json" in cb:
                        content_str = json.dumps(cb["json"], ensure_ascii=False)
                        break
                tool_results.append({"tool": tool_name, "content": content_str})

    return actual_tools, tool_results


def _extract_notes(answer: str) -> list[str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip().startswith(NOTE_MARKER)]
    return lines


def run_case(
    case: dict[str, Any],
    advisor: StrandsAdvisor,
    collector: EventCollector,
) -> CaseResult:
    trace_id = uuid.uuid4().hex
    thread_id = f"eval-{uuid.uuid4().hex}"
    answer = advisor.ask(case["question"], thread_id, trace_id)
    events = _events_for(collector, trace_id)

    # For StrandsAdvisor, tool calls/results live in agent.messages, not in log events
    agent = advisor._agents.get(thread_id)
    messages = agent.messages if agent is not None else []
    actual_tools, tool_results = _extract_tools_from_messages(messages)

    review_events = [event for event in events if event.get("event") == "output_review_summary"]
    review = review_events[-1] if review_events else {}
    router_events = [event for event in events if event.get("event") == "router_classified"]
    actual_intent = router_events[-1].get("intent", "") if router_events else ""
    router_fallbacks = [event for event in events if event.get("event") == "router_fallback_unsupported"]
    if router_fallbacks and not actual_intent:
        result_failure = f"\u610f\u56fe\u5206\u7c7b\u5931\u8d25\u5e76\u56de\u9000 unsupported\uff1a{router_fallbacks[-1].get('error', '\u672a\u77e5\u539f\u56e0')}"
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
    print(f"  \u610f\u56fe: {result.intent}" + (
        f"\uff08\u5b9e\u9645\u5206\u7c7b\uff1a{result.actual_intent}\uff09" if result.actual_intent and result.actual_intent != result.intent
        else ""
    ))
    print(f"  \u5de5\u5177: {result.actual_tools or '\u65e0'}")
    review = result.review or {}
    if review:
        print(f"  \u68c0\u67e5\u6e05\u5355: \u901a\u8fc7={review.get('passed')} \u8fd0\u884c={review.get('checks_run')} "
              f"\u5931\u8d25={len(review.get('failed_checks') or [])} \u4e0d\u53ef\u7528={len(review.get('unavailable_checks') or [])}")
    else:
        print("  \u68c0\u67e5\u6e05\u5355: \u672a\u6267\u884c\uff08\u56de\u7b54\u672a\u7ecf\u8fc7 output_review \u8282\u70b9\uff09")
    semantic = result.semantic_review
    if semantic:
        semantic_items = semantic.get("results") or []
        print(
            f"  \u8bed\u4e49\u8bc4\u5ba1: \u72b6\u6001={semantic.get('status')} "
            f"\u901a\u8fc7={sum(item.get('decision') == 'pass' for item in semantic_items)} "
            f"\u5931\u8d25={sum(item.get('decision') == 'fail' for item in semantic_items)} "
            f"\u4e0d\u786e\u5b9a={sum(item.get('decision') == 'uncertain' for item in semantic_items)}"
        )
        if semantic.get("model"):
            print(
                f"    Judge={semantic['model']} "
                f"prompt_version={semantic.get('prompt_version', 'unknown')}"
            )
        if semantic.get("error"):
            print(f"    \u4e0d\u53ef\u7528\u539f\u56e0\uff1a{semantic['error']}")
        for item in semantic_items:
            if item.get("decision") != "pass":
                evidence = item.get("evidence") or "\uff08\u672a\u627e\u5230\u76f4\u63a5\u8bc1\u636e\uff09"
                print(
                    f"    {item.get('id')} [{item.get('decision')}]: "
                    f"{item.get('reason', '')}\uff1b\u8bc1\u636e\uff1a{evidence}"
                )
    for failure in result.failures:
        print(f"  \u2717 {failure}")


def print_case_answer(result: CaseResult) -> None:
    print(f"  \u539f\u59cb\u95ee\u9898: {result.question}")
    print("  --- \u56de\u7b54 ---")
    print(result.answer)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation case set against the SA Advisor (Strands SDK).")
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
        print("\u7aef\u70b9\u8bca\u65ad\uff08\u9010\u9879\u52a0\u7801\u8bf7\u6c42\uff0c\u5b9a\u4f4d 5xx \u6765\u81ea\u54ea\u4e2a\u7279\u6027\uff09\uff1a")
        # Basic connectivity check in diagnose mode
        import httpx
        base_url = os.environ["LLM_BASE_URL"].rstrip("/")
        model = os.environ["LLM_MODEL"]
        url = base_url + "/chat/completions"
        headers = {
            "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
            "Content-Type": "application/json",
        }
        from pa_analytics import Store
        schema = [t.api_schema() for t in create_tools(Store("data/pa.db", "data/raw"))]
        probes = [
            ("A \u7eaf\u5bf9\u8bdd\uff0c\u65e0 tools", {"model": model, "messages": [{"role": "user", "content": "\u56de\u590d OK"}]}),
            ("B \u7eaf\u5bf9\u8bdd + max_tokens", {"model": model, "messages": [{"role": "user", "content": "\u56de\u590d OK"}], "max_tokens": 2048}),
            ("C \u77ed system + \u5de5\u5177 schema", {"model": model, "messages": [{"role": "user", "content": "\u67e5 2025-03 \u7684\u6708\u5ea6\u660e\u7ec6"}], "tools": schema, "max_tokens": 2048}),
        ]
        for label, payload in probes:
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=120)
                body = resp.text[:220].replace("\n", " ")
                print(f"  {resp.status_code}  {label}\n      {body}")
            except httpx.RequestError as exc:
                print(f"  EXC  {label}\n      {exc}")
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
    advisor = StrandsAdvisor(
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
            result = CaseResult(
                case_id=case["id"],
                name=case.get("name", case["id"]),
                intent=case.get("intent", ""),
                question=case["question"],
                answer="",
                semantic_review=(
                    {"status": "unavailable", "error": "\u6848\u4f8b\u8fd0\u884c\u5f02\u5e38\uff0c\u672a\u751f\u6210\u53ef\u8bc4\u5ba1\u56de\u7b54"}
                    if case.get("semantic_rubric") else {}
                ),
                failures=[f"\u8fd0\u884c\u5f02\u5e38\uff1a{exc}"],
            )
        results.append(result)
        print_case_report(result)
        if args.show_answer:
            print_case_answer(result)

    summary = summarize(results)
    print("\n" + "=" * 60)
    print(f"\u603b\u7528\u4f8b: {summary['total']}  \u901a\u8fc7: {summary['passed']}  \u901a\u8fc7\u7387: {summary['pass_rate']}%")
    for intent, bucket in sorted(summary["by_intent"].items()):
        rate = round(bucket["passed"] / bucket["total"] * 100, 1) if bucket["total"] else 0.0
        print(f"  {intent}: {bucket['passed']}/{bucket['total']} ({rate}%)")
    semantic_summary = summary["semantic_review"]
    routing_summary = summary["routing"]
    print(
        "\u8def\u7531\u5206\u7c7b: "
        f"\u6b63\u786e={routing_summary['correct']} \u9519\u8bef={routing_summary['incorrect']} "
        f"\u672a\u8bb0\u5f55={routing_summary['unrecorded']}"
    )
    if semantic_summary["cases"]:
        print(
            "\u8bed\u4e49\u8bc4\u5ba1\u6c47\u603b: "
            f"\u901a\u8fc7={semantic_summary['pass']} \u5931\u8d25={semantic_summary['fail']} "
            f"\u4e0d\u786e\u5b9a={semantic_summary['uncertain']} \u4e0d\u53ef\u7528={semantic_summary['unavailable']}"
        )
    if summary["failure_reasons"]:
        print("\u5931\u8d25\u539f\u56e0\u5206\u5e03:")
        for reason, count in summary["failure_reasons"].items():
            print(f"  {count:>3} x {reason}")
    if summary["derived_numbers"] or summary["unexplained_numbers"]:
        print(f"\u6570\u5b57\u6eaf\u6e90: \u6a21\u578b\u81ea\u884c\u63a8\u5bfc {summary['derived_numbers']} \u4e2a\uff0c"
              f"\u65e0\u6cd5\u89e3\u91ca\u6765\u6e90 {summary['unexplained_numbers']} \u4e2a")

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
        print(f"\n\u8be6\u7ec6\u7ed3\u679c\u5df2\u5199\u5165 {args.json_out}")

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
