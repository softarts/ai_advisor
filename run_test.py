#!/usr/bin/env python3
import subprocess
import sys

print("Running DQ03 test...")
result = subprocess.run(
    [sys.executable, "sa_advisor/eval/runner.py", "--case", "DQ03", "--show-answer"],
    capture_output=True,
    text=True,
    timeout=180,
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)
