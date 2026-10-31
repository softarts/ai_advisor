"""Tests for strands.tools.wrap_tools and make_strands_tool.

conftest.py (in this directory) bootstraps _strands_sdk before this module
is loaded, so ``_strands_sdk`` is available for the skipif check.
These tests do not require any LLM API keys.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock

import pytest

# Determine whether the SDK is available (conftest registered it)
strands_available = "_strands_sdk" in sys.modules

pytestmark = pytest.mark.skipif(
    not strands_available, reason="strands-agents not installed"
)


def _make_tool(name="get_test", description="A test tool", params=None, fn=None):
    """Create an advisor.tools.Tool with a mock function."""
    from advisor.tools import Tool

    if params is None:
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query"},
            },
            "required": ["query"],
        }
    if fn is None:
        fn = MagicMock(return_value={"status": "ok", "value": 42})

    return Tool(name=name, description=description, params_schema=params, fn=fn)


class TestMakeStrandsTool(unittest.TestCase):
    def test_wrapped_tool_has_correct_name(self):
        from strands.tools import make_strands_tool

        tool_obj = _make_tool(name="get_window_summary")
        wrapped = make_strands_tool(tool_obj)
        self.assertEqual(wrapped.tool_name, "get_window_summary")

    def test_wrapped_tool_has_correct_description(self):
        from strands.tools import make_strands_tool

        tool_obj = _make_tool(description="Returns window summary data")
        wrapped = make_strands_tool(tool_obj)
        self.assertIn("Returns window summary data", wrapped.tool_spec.get("description", ""))

    def test_wrapped_tool_spec_contains_input_schema(self):
        from strands.tools import make_strands_tool

        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        tool_obj = _make_tool(params=params)
        wrapped = make_strands_tool(tool_obj)
        schema = wrapped.tool_spec.get("inputSchema", {}).get("json", {})
        self.assertEqual(schema["properties"]["x"]["type"], "integer")


class TestWrapTools(unittest.TestCase):
    def test_wrap_tools_returns_list_of_correct_length(self):
        from strands.tools import wrap_tools

        tools = [_make_tool(f"tool_{i}") for i in range(3)]
        wrapped = wrap_tools(tools)
        self.assertEqual(len(wrapped), 3)

    def test_wrap_tools_names_match(self):
        from strands.tools import wrap_tools

        tools = [_make_tool("alpha"), _make_tool("beta")]
        wrapped = wrap_tools(tools)
        names = [w.tool_name for w in wrapped]
        self.assertIn("alpha", names)
        self.assertIn("beta", names)

    def test_wrapped_tool_calls_execute(self):
        """Calling the wrapped tool must delegate to advisor.tools.execute."""
        from strands.tools import make_strands_tool

        mock_fn = MagicMock(return_value={"status": "ok", "value": "result"})
        tool_obj = _make_tool(
            name="get_test",
            params={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=mock_fn,
        )
        wrapped = make_strands_tool(tool_obj)
        # Call the wrapped function directly (not via Strands tool-use protocol)
        result = wrapped(query="hello")
        mock_fn.assert_called_once_with(query="hello")

    def test_invalid_args_return_error_dict_not_exception(self):
        """execute() must catch validation errors and return an error dict."""
        from advisor.tools import execute

        tool_obj = _make_tool(
            name="strict_tool",
            params={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
        )
        # Call execute directly with wrong type — should not raise
        result = execute(tool_obj, {"n": "not-an-integer"})
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "error")
        self.assertIsNone(result.get("value"))

    def test_execute_is_not_bypassed(self):
        """The wrapped function must call execute(), not tool.fn() directly."""
        from strands.tools import make_strands_tool

        # Tool whose fn raises to prove execute() wraps the error
        def bad_fn(**kwargs):
            raise RuntimeError("direct fn call bypassed execute")

        tool_obj = _make_tool(
            name="bypass_test",
            params={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            fn=bad_fn,
        )
        wrapped = make_strands_tool(tool_obj)
        # execute() wraps RuntimeError into an error dict — no exception raised
        result = wrapped(q="test")
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "error")
