"""Wrap advisor.tools.Tool instances for use with the Strands Agent.

Uses ``_strands_sdk`` (the real strands-agents SDK) to avoid the name conflict
with this package's own directory named ``strands/``.

The Strands @tool decorator accepts an explicit ``inputSchema`` dict with a
``json`` key, which overrides the automatic schema inference from type hints.
This lets us forward each advisor Tool's existing JSON schema verbatim without
rewriting function signatures.
"""
from __future__ import annotations
from typing import Any

from _strands_sdk import tool as strands_tool
from advisor.tools import Tool, execute


def make_strands_tool(tool_obj: Tool):
    """Return a Strands-compatible DecoratedFunctionTool wrapping tool_obj.

    The returned object can be passed directly to ``Agent(tools=[...])``.
    """
    # Build the inputSchema that Strands expects: {"json": <JSON Schema object>}
    input_schema = {"json": tool_obj.params_schema}

    @strands_tool(
        name=tool_obj.name,
        description=tool_obj.description,
        inputSchema=input_schema,
    )
    def _fn(**kwargs: Any) -> Any:
        return execute(tool_obj, kwargs)

    # Set __name__ and __doc__ so unit tests can introspect them.
    _fn.__name__ = tool_obj.name  # type: ignore[attr-defined]
    _fn.__doc__ = tool_obj.description  # type: ignore[attr-defined]
    return _fn


def wrap_tools(advisor_tools: list[Tool]) -> list:
    """Convert a list of advisor Tool objects to Strands tool callables."""
    return [make_strands_tool(t) for t in advisor_tools]
