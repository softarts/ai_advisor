#!/usr/bin/env python3
"""Test script to verify the fixes."""
import sys
import os
from pathlib import Path

# Load .env
env_path = Path(".env")
if env_path.exists():
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

print("Environment loaded:")
for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
    v = os.environ.get(k, "NOT SET")
    print(f"  {k}={v[:30]}..." if len(v) > 30 else f"  {k}={v}")

# Test imports
print("\nTesting imports...")
try:
    from sa_advisor.model_provider import get_model, DEFAULT_TIMEOUT, MAX_RETRIES
    print(f"✓ model_provider imported (timeout={DEFAULT_TIMEOUT}, retries={MAX_RETRIES})")
except Exception as e:
    print(f"✗ model_provider import failed: {e}")
    sys.exit(1)

try:
    from eval.semantic_judge import _request_review
    print("✓ semantic_judge imported")
except Exception as e:
    print(f"✗ semantic_judge import failed: {e}")
    sys.exit(1)

try:
    from sa_advisor.agent import StrandsAdvisor
    print(f"✓ StrandsAdvisor imported (max_repair={StrandsAdvisor.MAX_REPAIR_ATTEMPTS}, timeout={StrandsAdvisor.MAX_EXECUTION_TIME}s)")
except Exception as e:
    print(f"✗ StrandsAdvisor import failed: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
print("\nNow running DQ03 test case...")

# Run the actual test
import subprocess
result = subprocess.run(
    [sys.executable, "sa_advisor/eval/runner.py", "--case", "DQ03", "--show-answer"],
    cwd=Path(__file__).parent,
    timeout=180,
)
sys.exit(result.returncode)
