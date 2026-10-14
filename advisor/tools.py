"""Tool definitions, schema validation, and adapters for portfolio analytics."""
from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from pa_analytics import Store, registry
from pa_analytics.metrics import MAX_CONTRIBUTORS

MAX_TOOL_RESULTS = MAX_CONTRIBUTORS


class Tool:
    def __init__(self, name: str, description: str, params_schema: dict, fn: Callable):
        self.name, self.description, self.params_schema, self.fn = name, description, params_schema, fn

    def api_schema(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                "parameters": self.params_schema}}


def _error(code: str, message: str, details: Any = None) -> dict:
    return {"value": None, "basis": "tool validation/execution", "period": [None, None],
            "currency": None, "status": "error", "warnings": [],
            "error": {"code": code, "message": message, **({"details": details} if details is not None else {})}}


def _validate(arguments: Any, schema: dict) -> tuple[dict | None, dict | None]:
    if not isinstance(arguments, dict):
        return None, _error("INVALID_ARGUMENTS", "arguments must be a JSON object")
    properties = schema.get("properties", {})
    extra = sorted(set(arguments) - set(properties))
    if extra and schema.get("additionalProperties") is False:
        return None, _error("UNKNOWN_ARGUMENT", "Unrecognized tool argument", extra)
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        return None, _error("MISSING_ARGUMENT", "Required tool argument is missing", missing)

    clean = dict(arguments)
    for key, value in arguments.items():
        rule = properties.get(key, {})
        typ = rule.get("type")
        valid = {"string": lambda v: isinstance(v, str),
                 "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                 "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                 "boolean": lambda v: isinstance(v, bool),
                 "object": lambda v: isinstance(v, dict),
                 "array": lambda v: isinstance(v, list)}.get(typ)
        if valid and not valid(value):
            return None, _error("INVALID_ARGUMENT_TYPE", f"{key} must be {typ}")
        if "enum" in rule and value not in rule["enum"]:
            return None, _error("INVALID_ARGUMENT", f"Unsupported value for {key}", {"allowed": rule["enum"]})
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                return None, _error("LIMIT_EXCEEDED", f"{key} is below its minimum", {"minimum": rule["minimum"]})
            if "maximum" in rule and value > rule["maximum"]:
                return None, _error("LIMIT_EXCEEDED", f"{key} exceeds its maximum", {"maximum": rule["maximum"]})
        if "pattern" in rule and isinstance(value, str) and not re.fullmatch(rule["pattern"], value):
            return None, _error("INVALID_ARGUMENT_FORMAT", f"{key} has an invalid format", {"pattern": rule["pattern"]})
        if rule.get("format") == "year-month" and isinstance(value, str):
            if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
                return None, _error("INVALID_ARGUMENT_FORMAT", f"{key} must use YYYY-MM")
            try:
                datetime.strptime(value, "%Y-%m")
            except ValueError:
                return None, _error("INVALID_ARGUMENT_FORMAT", f"{key} must use YYYY-MM")
    return clean, None


def create_tools(store: Store) -> list[Tool]:
    """Unified factory creating all Agent tools from registry backed by the data store."""
    result = []
    for spec in registry.TOOLS:
        schema = {**deepcopy(spec["parameters"]), "additionalProperties": False}
        if "window" in schema.get("properties", {}):
            schema["properties"]["window"] = {
                **schema["properties"]["window"], "default": "YTD"}
        if spec["name"] == "get_top_contributors":
            schema["properties"]["n"] = {**schema["properties"]["n"], "minimum": 1,
                                         "maximum": MAX_TOOL_RESULTS}

        def invoke(_name=spec["name"], **kwargs):
            return registry.call(store, _name, kwargs)

        result.append(Tool(spec["name"], spec["description"], schema, invoke))
    return result


def registry_tools(store: Store) -> list[Tool]:
    """Backward-compatible helper returning all tools except get_portfolio_detail."""
    return [t for t in create_tools(store) if t.name != "get_portfolio_detail"]


def portfolio_detail_tool(store: Store) -> Tool:
    """Backward-compatible helper returning get_portfolio_detail tool."""
    return next(t for t in create_tools(store) if t.name == "get_portfolio_detail")


def execute(tool: Tool, arguments: Any) -> dict:
    """Validate tool arguments and turn expected failures into machine-readable results."""
    if not isinstance(tool, Tool):
        return _error("UNKNOWN_TOOL", "Tool is not registered")
    clean, error = _validate(arguments, tool.params_schema)
    if error:
        return error
    try:
        value = tool.fn(**clean)
        if isinstance(value, dict) and "status" in value:
            return value
        return {"value": value, "basis": "tool result", "period": [None, None], "currency": None,
                "status": "ok", "warnings": []}
    except KeyError as exc:
        return _error("UNKNOWN_VALUE", "A requested value is not supported", str(exc))
    except (TypeError, ValueError) as exc:
        return _error("INVALID_ARGUMENT", str(exc))
    except Exception as exc:
        return _error("TOOL_EXECUTION_FAILED", "Tool execution failed", str(exc))
