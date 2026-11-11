"""pytest configuration for sa_advisor tests.

sa_advisor/ uses the strands package directly (no name-shadowing).
If strands-agents is installed in the active environment this file is a no-op.
If it lives in a non-default venv, set STRANDS_SITE_PACKAGES in the environment
to point at the site-packages directory and this conftest will add it to sys.path.

Example (.env or environment):
    STRANDS_SITE_PACKAGES=/path/to/.venv/lib/python3.x/site-packages
"""
import os
import sys

_extra = os.environ.get("STRANDS_SITE_PACKAGES", "")
if _extra and _extra not in sys.path:
    sys.path.insert(0, _extra)

# Verify the SDK is importable so tests get a clear message instead of a
# confusing ImportError deep inside a test file.
try:
    import strands  # noqa: F401
    _strands_available = True
except ImportError:
    _strands_available = False

import pytest

def pytest_collection_modifyitems(items):
    if not _strands_available:
        skip = pytest.mark.skip(reason="strands-agents not installed; set STRANDS_SITE_PACKAGES")
        for item in items:
            item.add_marker(skip)
