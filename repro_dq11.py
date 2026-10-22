"""Offline reproduction of the DQ11 evaluation failures (no LLM calls)."""
import json
import re
import sys
from pathlib import Path

from pa_analytics import Store
from advisor.tools import create_tools
from advisor.output_review import _strip_prior_annotation, ungrounded_numbers
from eval.report import _content_matches

ATT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"c:\Users\wirel\AppData\Roaming\Code\agentSessionData\fc26bb06-3999-47dc-b5d2-68ae46a2c3f5"
    r"\attachments\79c0a88f-294e-42d5-9560-1c649d660411\Pasted text #1.txt"
)

text = ATT.read_text(encoding="utf-8")
marker = "--- " + "\u56de\u7b54" + " ---"
answer = text.split(marker, 1)[1]
answer = answer.split("\n\n========", 1)[0]
answer = "\n".join(l for l in answer.splitlines() if not l.strip().startswith("\u26a0\ufe0f"))
candidate = _strip_prior_annotation(answer)

store = Store("data/pa.db", "data/raw")
tools = {t.name: t for t in create_tools(store)}
result = tools["get_month_detail"].fn(month="2025-03")
tool_results = [{"tool": "get_month_detail", "content": json.dumps(result, ensure_ascii=False)}]

out = []
out.append("=== ungrounded numbers ===")
out.append(str(ungrounded_numbers(candidate, tool_results)))

out.append("")
out.append("=== rank block in tool result ===")
out.append(json.dumps(result.get("return", {}).get("rank", {}), ensure_ascii=False))

out.append("")
out.append("=== must_include checks ===")
must = ["4.49", "-5.63", "18,935", "60,000", "3.3", "4,470",
        ["\u5165\u91d1\u4e0d\u662f\u6536\u76ca", "\u5165\u91d1\u4e0d\u8ba1\u5165\u6536\u76ca"],
        "K3", "K13"]
for tok in must:
    out.append("%r -> %s" % (tok, _content_matches(tok, candidate)))

out.append("")
out.append("=== occurrences of the deposit phrase ===")
needle = "\u5165\u91d1"
for m in re.finditer(needle, candidate):
    out.append(repr(candidate[max(0, m.start() - 40):m.start() + 50]))

out.append("")
out.append("=== bold numbered section headings ===")
for line in candidate.splitlines():
    if re.match(r"^\s*\*\*\d", line):
        out.append(repr(line[:70]))

Path("repro_out.txt").write_text("\n".join(out), encoding="utf-8")
print("written repro_out.txt")
