#!/usr/bin/env python3
"""Simple import test."""
import sys
import os
from pathlib import Path

# Load environment variables
env_path = Path('.env')
if env_path.exists():
    print(f"Loading .env from {env_path.absolute()}")
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))
    print("Environment loaded")
else:
    print(".env not found")

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print("\nAttempting to run sa_advisor/eval/runner.py...")

try:
    # Import the runner module directly
    sys.argv = ['runner.py', '--case', 'DQ03', '--show-answer']
    exec(open('sa_advisor/eval/runner.py').read(), {'__name__': '__main__'})
except SystemExit as e:
    print(f"\nSystemExit with code: {e.code}")
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
