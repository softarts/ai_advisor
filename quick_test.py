import sys
import os
from pathlib import Path

env_path = Path('.env')
if env_path.exists():
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))

print('LLM_BASE_URL:', os.environ.get('LLM_BASE_URL', 'NOT SET'))

sys.path.insert(0, '.')
from sa_advisor import model_provider
print('Module imported successfully')
print('DEFAULT_TIMEOUT:', model_provider.DEFAULT_TIMEOUT)
print('MAX_RETRIES:', model_provider.MAX_RETRIES)
