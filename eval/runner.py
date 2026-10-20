#!/usr/bin/env python3
"""Evaluation runner for the real advisor agent.

Runs a fixed case set against the live agent (real LLM + real tools + real
answer checklist) and decides every outcome in code. Each case runs in its own
thread so no context leaks between cases.
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
from advisor.output_review import CheckContext
from advisor.tools import create_tools
from eval.report import CaseResult, evaluate_case, summarize

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
        "max_tokens": 16,
    }
    headers = {
        "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=60)
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
        raise RuntimeError(
            f"端点 {url} 返回 {response.status_code}：{response.text[:300]}"
        )
    return f"端点与模型可用：{model} @ {url}"


def load_cases(case_filter: list[str] | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        if path.name == "why_month.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
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
    for failure in result.failures:
        print(f"  ✗ {failure}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation case set against the live agent.")
    parser.add_argument("--case", action="append", help="Run only the given case id (repeatable)")
    parser.add_argument("--db", default="data/pa.db")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--json-out", help="Write the detailed results as JSON to this path")
    parser.add_argument("--show-answer", action="store_true", help="Print the full answer of every case")
    args = parser.parse_args(argv)

    load_env_from_project_root()
    ensure_llm_settings()
    print(preflight_llm())

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
            result = CaseResult(
                case_id=case["id"],
                name=case.get("name", case["id"]),
                intent=case.get("intent", ""),
                question=case["question"],
                answer="",
                failures=[f"运行异常：{exc}"],
            )
        results.append(result)
        print_case_report(result)
        if args.show_answer:
            print("  --- 回答 ---")
            print(result.answer)

    summary = summarize(results)
    print("\n" + "=" * 60)
    print(f"总用例: {summary['total']}  通过: {summary['passed']}  通过率: {summary['pass_rate']}%")
    for intent, bucket in sorted(summary["by_intent"].items()):
        rate = round(bucket["passed"] / bucket["total"] * 100, 1) if bucket["total"] else 0.0
        print(f"  {intent}: {bucket['passed']}/{bucket['total']} ({rate}%)")
    if summary["failure_reasons"]:
        print("失败原因分布:")
        for reason, count in summary["failure_reasons"].items():
            print(f"  {count:>3} x {reason}")

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
