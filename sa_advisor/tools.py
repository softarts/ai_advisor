"""Wrap common.tools.Tool instances for use with the Strands Agent.

The Strands @tool decorator accepts an explicit ``inputSchema`` dict with a
``json`` key, which overrides automatic schema inference from type hints.
This lets us forward each Tool's existing JSON schema verbatim without
rewriting function signatures.
"""
from __future__ import annotations
from typing import Any

from strands import tool as strands_tool

from common.tools import Tool, execute


def make_strands_tool(tool_obj: Tool):
    """Return a Strands-compatible DecoratedFunctionTool wrapping tool_obj."""
    input_schema = {"json": tool_obj.params_schema}

    @strands_tool(
        name=tool_obj.name,
        description=tool_obj.description,
        inputSchema=input_schema,
    )
    def _fn(**kwargs: Any) -> Any:
        return execute(tool_obj, kwargs)

    _fn.__name__ = tool_obj.name  # type: ignore[attr-defined]
    _fn.__doc__ = tool_obj.description  # type: ignore[attr-defined]
    return _fn


def wrap_tools(advisor_tools: list[Tool]) -> list:
    """Convert a list of Tool objects to Strands tool callables."""
    return [make_strands_tool(t) for t in advisor_tools]
