#!/usr/bin/env python3
"""Test script to check imports."""
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

print("Testing imports...")

# Test 1: model_provider
print("\n1. Testing model_provider...")
try:
    from sa_advisor.model_provider import get_model, DEFAULT_TIMEOUT, MAX_RETRIES
    print(f"   ✓ Success (timeout={DEFAULT_TIMEOUT}, retries={MAX_RETRIES})")
except Exception as e:
    print(f"   ✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 2: semantic_judge
print("\n2. Testing semantic_judge...")
try:
    from eval.semantic_judge import _request_review
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 3: agent
print("\n3. Testing StrandsAdvisor...")
try:
    from sa_advisor.agent import StrandsAdvisor
    print(f"   ✓ Success (max_repair={StrandsAdvisor.MAX_REPAIR_ATTEMPTS}, timeout={StrandsAdvisor.MAX_EXECUTION_TIME}s)")
except Exception as e:
    print(f"   ✗ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
