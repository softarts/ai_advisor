"""Agent-library-agnostic shared modules.

Provides:
  - tools          : Tool dataclass, execute(), create_tools()
  - output_review  : answer-quality checkers (numeric provenance, product policy)
  - checklist_config: per-intent check lists and definitions
  - llm_settings   : shared token budgets and timeouts
  - prompts        : intent profiles, compose-template loader, fixed-reply helpers
"""
