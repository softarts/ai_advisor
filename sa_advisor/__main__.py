"""Entry point: python -m sa_advisor

Strands-SDK-based portfolio advisor REPL.
No _bootstrap_sdk hackery needed: sa_advisor/ does not shadow the strands package.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from pa_analytics import Store
from common.tools import create_tools
from .agent import StrandsAdvisor


def _load_env() -> None:
    path = Path(".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


COMMANDS = [
    "/import-report",
    "/history",
    "/clear",
    "/remember",
    "/recall",
    "/help",
    "/exit",
    "/quit",
]


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for cmd in COMMANDS:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))


def _make_key_bindings() -> KeyBindings:
    kb = KeyBindings()
    state = {"last_ctrl_c": 0.0}

    @kb.add("c-c")
    def _(event):
        now = time.monotonic()
        if now - state["last_ctrl_c"] < 1.0:
            event.app.exit(exception=EOFError)
        else:
            state["last_ctrl_c"] = now
            event.app.output.write("\n再按一次 Ctrl-C 退出（或输入 /exit）\n")
            event.app.current_buffer.reset()

    return kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Strands AI Advisor REPL")
    parser.add_argument("--db", default="data/pa.db")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--user-id", default="default")
    args = parser.parse_args()

    _load_env()

    logdir = Path("logs")
    logdir.mkdir(exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logdir / f"sa_advisor_{now_str}_{uuid.uuid4().hex[:8]}.jsonl"
    logger = logging.getLogger("sa_advisor")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    store = Store(args.db, args.raw_dir)

    advisor: StrandsAdvisor | None = None
    if all(os.environ.get(k) for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")):
        tools = create_tools(store)
        advisor = StrandsAdvisor(
            tools,
            logger,
            args.user_id,
            timezone_name=os.environ.get("APP_TIMEZONE", "Asia/Singapore"),
            data_as_of=lambda: store.current().period_end,
        )

    thread_id = uuid.uuid4().hex
    history_path = Path("logs") / ".sa_advisor_history"
    session = PromptSession(
        history=FileHistory(str(history_path)),
        completer=CommandCompleter(),
        complete_while_typing=True,
        key_bindings=_make_key_bindings(),
    )
    print(f"SA Advisor (Strands) · 日志 {log_path} · 输入 /help 查看命令")

    while True:
        try:
            line = session.prompt("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break

        if line == "/help":
            print(
                "/import-report [文件路径] 导入报告；"
                "/history 本轮记忆；"
                "/clear 清空本轮；"
                "/remember 键=值 保存长期记忆；"
                "/recall 查看长期记忆；"
                "/exit 退出"
            )
            continue

        if line.startswith("/import-report"):
            path_arg = line[len("/import-report"):].strip()
            result = store.ingest(path_arg)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue

        if line == "/clear":
            thread_id = uuid.uuid4().hex
            print("已清空本轮对话记忆")
            continue

        if line == "/history":
            if advisor and thread_id in advisor._agents:
                msgs = advisor._agents[thread_id].messages
                for m in msgs:
                    role = m.get("role", "?")
                    for block in m.get("content", []):
                        if "text" in block:
                            print(f"{role}: {block['text'][:200]}")
            else:
                print("暂无对话")
            continue

        if line.startswith("/remember "):
            key, sep, value = line[len("/remember "):].partition("=")
            if sep and key.strip():
                if advisor:
                    advisor.remember(key.strip(), value.strip())
                logger.info(json.dumps({
                    "event": "memory_operation",
                    "operation": "put",
                    "user_id": args.user_id,
                    "key": key.strip(),
                }, ensure_ascii=False))
                print("已保存到本次运行的长期记忆")
            else:
                print("格式：/remember 键=值")
            continue

        if line == "/recall":
            if advisor:
                print(json.dumps(advisor.recall(), ensure_ascii=False, indent=2))
            else:
                print("{}")
            logger.info(json.dumps({
                "event": "memory_operation",
                "operation": "search",
                "user_id": args.user_id,
            }, ensure_ascii=False))
            continue

        if not advisor:
            print("请在环境变量或 .env 中设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL")
            continue

        trace = uuid.uuid4().hex
        logger.info(json.dumps({
            "event": "user_input",
            "trace_id": trace,
            "input": line,
        }, ensure_ascii=False))
        try:
            answer = advisor.ask(line, thread_id, trace)
            logger.info(json.dumps({
                "event": "final_response",
                "trace_id": trace,
                "response": answer,
            }, ensure_ascii=False))
            print(f"AI> {answer}")
        except Exception as e:
            logger.error(json.dumps({
                "event": "error",
                "trace_id": trace,
                "error": str(e),
            }, ensure_ascii=False))
            print(f"调用失败：{e}")


if __name__ == "__main__":
    main()
