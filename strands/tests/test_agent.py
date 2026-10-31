"""Tests for strands.agent.StrandsAdvisor.

conftest.py (in this directory) bootstraps _strands_sdk before this module
loads.  No real LLM calls are made; all model and router interactions are
mocked.
"""
from __future__ import annotations

import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Determine whether the SDK is available (conftest registered it)
strands_available = "_strands_sdk" in sys.modules

pytestmark = pytest.mark.skipif(
    not strands_available, reason="strands-agents not installed"
)


def _logger():
    return logging.getLogger("test_agent")


def _make_advisor(tools=None):
    from strands.agent import StrandsAdvisor

    return StrandsAdvisor(tools or [], _logger(), user_id="test-user")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_agent_result(text: str):
    """Return a minimal object whose str() is ``text``."""
    result = MagicMock()
    result.__str__ = MagicMock(return_value=text)
    return result


def _mock_agent_instance(text: str = "mock answer"):
    mock_agent = MagicMock()
    mock_agent.return_value = _fake_agent_result(text)
    mock_agent.messages = []
    return mock_agent


# ---------------------------------------------------------------------------
# Unsupported intent
# ---------------------------------------------------------------------------

class TestUnsupportedIntent(unittest.TestCase):
    @patch("strands.agent.classify_intent", return_value="unsupported")
    @patch("strands.agent.load_compose_template", return_value="---\n---\n# 标题\n固定回复内容。")
    @patch("strands.agent.get_model")
    def test_unsupported_returns_fixed_reply(self, mock_model, mock_template, mock_classify):
        advisor = _make_advisor()
        with patch("strands.agent.Agent") as mock_agent_cls:
            result = advisor.ask("随便问个不支持的问题", thread_id="t1")
        # Agent constructor must NOT have been called
        mock_agent_cls.assert_not_called()
        self.assertEqual(result, "固定回复内容。")

    @patch("strands.agent.classify_intent", return_value="unsupported")
    @patch("strands.agent.load_compose_template", return_value="")
    @patch("strands.agent.get_model")
    def test_unsupported_with_empty_template_falls_back(self, mock_model, mock_tmpl, mock_classify):
        advisor = _make_advisor()
        with patch("strands.agent.Agent"):
            result = advisor.ask("问题", thread_id="t1")
        self.assertIn("超出了本系统的服务范围", result)


# ---------------------------------------------------------------------------
# data_query intent
# ---------------------------------------------------------------------------

class TestDataQueryIntent(unittest.TestCase):
    @patch("strands.agent.classify_intent", return_value="data_query")
    @patch("strands.agent.get_model")
    @patch("strands.agent.wrap_tools", return_value=[])
    def test_data_query_calls_agent(self, mock_wrap, mock_model, mock_classify):
        advisor = _make_advisor()
        mock_inst = _mock_agent_instance("数据回答。")

        with patch("strands.agent.Agent", return_value=mock_inst) as mock_cls:
            result = advisor.ask("账户今年收益？", thread_id="t2")

        mock_cls.assert_called_once()
        mock_inst.assert_called_once_with("账户今年收益？")
        self.assertEqual(result, "数据回答。")

    @patch("strands.agent.classify_intent", return_value="data_query")
    @patch("strands.agent.get_model")
    @patch("strands.agent.wrap_tools", return_value=[])
    def test_same_thread_reuses_agent(self, mock_wrap, mock_model, mock_classify):
        advisor = _make_advisor()
        mock_inst = _mock_agent_instance("ok")

        with patch("strands.agent.Agent", return_value=mock_inst) as mock_cls:
            advisor.ask("第一问", thread_id="same-thread")
            advisor.ask("第二问", thread_id="same-thread")

        # Agent() constructor called only once for the same thread
        self.assertEqual(mock_cls.call_count, 1)

    @patch("strands.agent.classify_intent", return_value="data_query")
    @patch("strands.agent.get_model")
    @patch("strands.agent.wrap_tools", return_value=[])
    def test_different_threads_get_separate_agents(self, mock_wrap, mock_model, mock_classify):
        advisor = _make_advisor()
        mock_a = _mock_agent_instance("回答 A")
        mock_b = _mock_agent_instance("回答 B")

        with patch("strands.agent.Agent", side_effect=[mock_a, mock_b]) as mock_cls:
            advisor.ask("问题", thread_id="thread-A")
            advisor.ask("问题", thread_id="thread-B")

        self.assertEqual(mock_cls.call_count, 2)


# ---------------------------------------------------------------------------
# product_info — tool filtering
# ---------------------------------------------------------------------------

class TestProductInfoToolFiltering(unittest.TestCase):
    @patch("strands.agent.classify_intent", return_value="product_info")
    @patch("strands.agent.get_model")
    def test_product_info_filters_to_get_capabilities(self, mock_model, mock_classify):
        from advisor.tools import Tool

        caps_tool = Tool(
            "get_capabilities",
            "Describes supported capabilities",
            {"type": "object", "properties": {}},
            MagicMock(return_value={"status": "ok", "value": []}),
        )
        data_tool = Tool(
            "get_window_summary",
            "Returns window summary",
            {"type": "object", "properties": {"window": {"type": "string"}}},
            MagicMock(return_value={"status": "ok", "value": {}}),
        )

        advisor = _make_advisor([caps_tool, data_tool])
        mock_inst = _mock_agent_instance("产品回答。")

        with patch("strands.agent.Agent", return_value=mock_inst) as mock_cls:
            advisor.ask("你支持什么功能？", thread_id="pi-thread")

        call_kwargs = mock_cls.call_args[1]  # keyword args
        passed_tools = call_kwargs.get("tools", [])
        tool_names = [t.tool_name for t in passed_tools]
        self.assertIn("get_capabilities", tool_names)
        self.assertNotIn("get_window_summary", tool_names)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestMemory(unittest.TestCase):
    def test_remember_and_recall(self):
        advisor = _make_advisor()
        advisor.remember("risk", "conservative")
        advisor.remember("currency", "SGD")
        mem = advisor.recall()
        self.assertEqual(mem["risk"], "conservative")
        self.assertEqual(mem["currency"], "SGD")

    def test_recall_empty_by_default(self):
        advisor = _make_advisor()
        self.assertEqual(advisor.recall(), {})
