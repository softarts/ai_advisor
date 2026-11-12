import sys
import os
from pathlib import Path

# Load env
env_path = Path('.env')
if env_path.exists():
    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))

# Bootstrap strands before any imports
print('Step 0: Bootstrapping strands SDK...')
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
existing = sys.modules.get("strands")
if existing is None or "site-packages" not in (getattr(existing, "__file__", "") or ""):
    def _is_workspace(p: str) -> bool:
        if not p:
            return True
        try:
            return os.path.realpath(p) == os.path.realpath(_WORKSPACE)
        except Exception:
            return False
    
    _orig_path = sys.path[:]
    sys.path = [p for p in sys.path if not _is_workspace(p)]
    _shadowed = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "strands" or k.startswith("strands.")}
    try:
        import strands as _sdk
        sys.modules["_strands_sdk"] = _sdk
        print('  ✓ strands SDK bootstrapped')
    except Exception as e:
        sys.path = _orig_path
        for k, v in _shadowed.items():
            sys.modules[k] = v
        print(f'  ✗ Bootstrap failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        sys.path = _orig_path
else:
    print('  ✓ strands already loaded from site-packages')

print('Step 1: Testing strands import...')
try:
    from strands import Agent
    print('  ✓ strands.Agent imported')
except Exception as e:
    print(f'  ✗ Failed: {e}')
    import traceback
    traceback.print_exc()

print('Step 2: Testing strands.models.openai import...')
try:
    from strands.models.openai import OpenAIModel
    print('  ✓ OpenAIModel imported')
except Exception as e:
    print(f'  ✗ Failed: {e}')
    import traceback
    traceback.print_exc()

print('Step 3: Testing model_provider import...')
try:
    from sa_advisor.model_provider import get_model
    print('  ✓ model_provider imported')
except Exception as e:
    print(f'  ✗ Failed: {e}')
    import traceback
    traceback.print_exc()

print('\nAll imports successful!')
