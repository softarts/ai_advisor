#!/Users/zhourui/work/enterpriseAI-knowledge/.venv-py314/bin/python3
"""Smoke test for router + agent graph (no LLM calls)."""
import sys
sys.path.insert(0, "/Users/zhourui/work/ai_advisor")

from advisor.router import INTENT_PROFILES, load_compose_template
import logging

# 1. Template loading
for intent in ("data_query", "product_info", "unsupported"):
    t = load_compose_template(intent)
    assert t, f"empty template for {intent}"
    print(f"{intent}: {len(t)} chars, has version header: {'version' in t}")

# 2. fixed_reply text extraction
tmpl = load_compose_template("unsupported")
lines = tmpl.splitlines()
reply_lines = []
in_front_matter = False
past_front_matter = False
for line in lines:
    if line.strip() == "---":
        if not past_front_matter:
            in_front_matter = not in_front_matter
            if not in_front_matter:
                past_front_matter = True
        continue
    if not past_front_matter:
        continue
    if not reply_lines:
        if line.startswith("#") or not line.strip():
            continue
    reply_lines.append(line)
reply = "\n".join(reply_lines).strip()
print("fixed_reply text:", repr(reply[:80]))
assert "超出了本系统的服务范围" in reply, "fixed reply text missing"

# 3. Placeholder checks
dq = load_compose_template("data_query")
assert "{{total_complete_months}}" in dq, "placeholder missing"
assert "{{example_month}}" in dq, "example_month placeholder missing"
assert "110" not in dq, "hardcoded 110 found"
assert "2025-03" not in dq, "hardcoded 2025-03 found"
print("data_query placeholders OK")

# 4. Agent graph structure
from advisor.agent import Advisor, State
from pa_analytics import Store
from advisor.tools import create_tools

fields = State.__annotations__
assert "intent" in fields, "intent field missing from State"
print("State fields:", list(fields.keys()))

s = Store("data/pa.db")
tools = create_tools(s)
logger = logging.getLogger("test")
adv = Advisor(tools, logger, data_as_of="2026-10-09")
nodes = list(adv.graph.nodes)
print("Graph nodes:", nodes)
assert "router" in nodes
assert "llm" in nodes
assert "fixed_reply" in nodes
assert "tools" in nodes

# 5. profile allowed_tools integrity
from advisor.router import INTENT_PROFILES
all_tool_names = {t.name for t in tools}
for intent, profile in INTENT_PROFILES.items():
    unknown = set(profile["allowed_tools"]) - all_tool_names
    assert not unknown, f"intent {intent} references unknown tools: {unknown}"
    print(f"{intent}: {len(profile['allowed_tools'])} allowed tools OK")

print("\nAll smoke checks passed.")

