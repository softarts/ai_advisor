"""Pre-load the real strands-agents SDK before any test module imports it.

Our package directory is named ``strands/``, which shadows the installed
strands-agents SDK when the workspace root is on sys.path.  This conftest
module runs first (pytest loads conftest.py before test files) and ensures:

  1. The real SDK is importable as ``_strands_sdk``.
  2. Our package submodules (strands.agent, strands.tools, etc.) are loaded
     and registered in sys.modules under their correct names AFTER the SDK,
     so patch("strands.agent.xxx") targets our code, not the SDK's.
"""
import os
import sys
import importlib.util

_WORKSPACE = "/Users/zhourui/work/ai_advisor"
_VENV_SITE = (
    "/Users/zhourui/work/enterpriseAI-knowledge/.venv-py314"
    "/lib/python3.14/site-packages"
)


def _is_workspace(p: str) -> bool:
    if not p:  # empty string = cwd
        return True
    try:
        return os.path.realpath(p) == os.path.realpath(_WORKSPACE)
    except Exception:
        return False


def _ensure_sdk() -> None:
    """Bootstrap the SDK alias and register our submodules."""
    if "_strands_sdk" in sys.modules:
        return

    # ------------------------------------------------------------------ #
    # Step 1: Load the real SDK while the workspace is off sys.path       #
    # ------------------------------------------------------------------ #
    _orig_path = sys.path[:]
    sys.path = [p for p in sys.path if not _is_workspace(p)]
    if _VENV_SITE not in sys.path:
        sys.path.insert(0, _VENV_SITE)

    # Save any partial registration of our package
    _shadowed = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "strands" or k.startswith("strands.")}

    try:
        import strands as _sdk  # the real SDK
    except Exception as exc:
        # Restore and give up
        sys.path = _orig_path
        for k, v in _shadowed.items():
            sys.modules[k] = v
        return

    # Register the real SDK under a stable alias
    sys.modules["_strands_sdk"] = _sdk
    for k in list(sys.modules):
        if k == "strands" or k.startswith("strands."):
            alias = "_strands_sdk" + k[len("strands"):]
            sys.modules.setdefault(alias, sys.modules[k])

    # ------------------------------------------------------------------ #
    # Step 2: Remove SDK's strands.* submodules so ours can register      #
    # ------------------------------------------------------------------ #
    for k in [k for k in sys.modules if k.startswith("strands.")]:
        del sys.modules[k]

    # ------------------------------------------------------------------ #
    # Step 3: Restore sys.path + our package                              #
    # ------------------------------------------------------------------ #
    sys.path = _orig_path
    for k, v in _shadowed.items():
        sys.modules[k] = v

    # Ensure our strands package __init__ is registered correctly
    strands_pkg = sys.modules.get("strands")
    if strands_pkg is None or getattr(strands_pkg, "__file__", "") != (
        f"{_WORKSPACE}/strands/__init__.py"
    ):
        spec = importlib.util.spec_from_file_location(
            "strands",
            f"{_WORKSPACE}/strands/__init__.py",
            submodule_search_locations=[f"{_WORKSPACE}/strands"],
        )
        pkg = importlib.util.module_from_spec(spec)
        pkg.__package__ = "strands"
        pkg.__path__ = [f"{_WORKSPACE}/strands"]
        sys.modules["strands"] = pkg
        spec.loader.exec_module(pkg)

    # ------------------------------------------------------------------ #
    # Step 4: Load our submodules in dependency order                     #
    # ------------------------------------------------------------------ #
    _submodules = [
        ("strands.prompts",        f"{_WORKSPACE}/strands/prompts.py"),
        ("strands.model_provider", f"{_WORKSPACE}/strands/model_provider.py"),
        ("strands.tools",          f"{_WORKSPACE}/strands/tools.py"),
        ("strands.agent",          f"{_WORKSPACE}/strands/agent.py"),
    ]
    for mod_name, mod_path in _submodules:
        # Skip if already loaded from our workspace
        existing = sys.modules.get(mod_name)
        if existing is not None and _WORKSPACE in (getattr(existing, "__file__", "") or ""):
            continue
        # Remove stale entry (could be SDK's or None)
        sys.modules.pop(mod_name, None)
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "strands"
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)


_ensure_sdk()
