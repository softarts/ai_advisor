"""Wrap common.tools.Tool instances for use with the Strands Agent.

Strands validates tool input against the function's *parameter signature*, not
against a raw JSON schema.  We therefore build a proper Python signature for
each Tool by reading its JSON Schema properties, then apply @tool so that
Strands can introspect parameters correctly and pass them as keyword arguments.

Thread safety: Strands executes tools in worker threads via asyncio.to_thread().
SQLite connections are not thread-safe, so each tool function re-opens the Store
(and its underlying SQLite connection) in the calling thread rather than sharing
the Store instance created in the main thread.
"""
from __future__ import annotations

import inspect
from typing import Any

from strands import tool as strands_tool

from pa_analytics import Store
from common.tools import Tool, execute, create_tools


def _make_tool_fn(tool_obj: Tool, db_path: str, raw_dir: str):
    """Return a plain Python function whose signature matches the Tool's schema.

    Strands' ``validate_input`` builds a Pydantic model from the function's
    *parameter names and defaults*.  Using ``**kwargs`` would collapse all
    parameters into a single required ``kwargs`` field and cause validation
    failures.  Generating an explicit ``inspect.Signature`` gives each
    parameter its own name and default value so validation and dispatch work
    correctly.

    Each invocation creates a fresh Store (new SQLite connection) so the tool
    is safe to call from any thread (Strands uses asyncio.to_thread internally).
    """
    props = tool_obj.params_schema.get("properties", {})
    required_keys = set(tool_obj.params_schema.get("required", []))
    defaults = {k: v.get("default") for k, v in props.items() if "default" in v}
    tool_name = tool_obj.name

    parameters = []
    for name in props:
        if name in required_keys:
            parameters.append(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation=Any)
            )
        else:
            parameters.append(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation=Any, default=defaults.get(name))
            )

    def _fn(**kwargs: Any) -> Any:
        # Re-create Store in this thread to avoid SQLite cross-thread errors.
        thread_store = Store(db_path, raw_dir)
        thread_tools = {t.name: t for t in create_tools(thread_store)}
        thread_tool = thread_tools[tool_name]
        return execute(thread_tool, kwargs)

    _fn.__name__ = tool_obj.name
    _fn.__doc__ = tool_obj.description
    _fn.__signature__ = inspect.Signature(parameters)
    return _fn


def make_strands_tool(tool_obj: Tool, db_path: str, raw_dir: str):
    """Return a Strands DecoratedFunctionTool wrapping tool_obj."""
    return strands_tool(_make_tool_fn(tool_obj, db_path, raw_dir))


def wrap_tools(advisor_tools: list[Tool], db_path: str, raw_dir: str) -> list:
    """Convert a list of Tool objects to Strands tool callables."""
    return [make_strands_tool(t, db_path, raw_dir) for t in advisor_tools]
