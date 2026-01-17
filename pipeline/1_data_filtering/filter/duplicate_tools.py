from typing import List, Dict, Any, Tuple
import json


def _strip_descriptions(obj: Any) -> Any:
    """
    Recursively remove all 'description' fields from a JSON-like structure.
    """
    if isinstance(obj, dict):
        return {k: _strip_descriptions(v) for k, v in obj.items() if k != "description"}
    if isinstance(obj, list):
        return [_strip_descriptions(item) for item in obj]
    return obj


def _canonicalize_parameters(parameters: Dict[str, Any]) -> Any:
    """
    Produce a canonical, comparable representation of a JSON schema-like dict.

    We use json.dumps with sort_keys=True after normalizing empty structures so
    that logically equivalent schemas compare equal.
    """
    if parameters is None or parameters == {}:
        # Treat missing/empty as an empty object schema
        parameters = {"type": "object", "properties": {}, "required": []}
    # Remove all description fields before comparison
    parameters = _strip_descriptions(parameters)
    return json.dumps(parameters, sort_keys=True, ensure_ascii=False)


def validate_no_duplicate_tools_by_schema(tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate that there are no duplicate tools having the same function name and identical schema.

    A duplicate is defined as another tool where:
      - tool["type"] == "function"
      - function.name is equal, and
      - function.parameters (after canonicalization) are equal

    Returns (True, "") if no duplicates; otherwise (False, error_message).
    """
    seen = {}
    for idx, tool in enumerate(tools):
        if tool.get("type") != "function" or "function" not in tool:
            raise ValueError(f"Tool {tool} is not a valid function")
            # Ignore non-function entries; other validators handle schema correctness
            continue
        func = tool.get("function", {})
        name = func.get("name")
        if not name:
            raise ValueError(f"Tool {tool} is not a valid function")
            # Skip invalid entries; other validators will flag
            continue
        parameters = func.get("parameters")
        key = (name, _canonicalize_parameters(parameters))
        if key in seen:
            prev_idx = seen[key]
            return False, (
                f"Duplicate tool detected: name '{name}' with identical schema appears at indices {prev_idx} and {idx}"
            )
        seen[key] = idx
    return True, ""


def validate_no_duplicate_tools_by_name(tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate that there are no duplicate tools having the same function name and identical schema.

    A duplicate is defined as another tool where:
      - tool["type"] == "function"
      - function.name is equal, and
      - function.parameters (after canonicalization) are equal

    Returns (True, "") if no duplicates; otherwise (False, error_message).
    """
    seen = set()
    for idx, tool in enumerate(tools):
        if tool.get("type") != "function" or "function" not in tool:
            raise ValueError(f"Tool {tool} is not a valid function")
            # Ignore non-function entries; other validators handle schema correctness
            continue
        func = tool.get("function", {})
        name = func.get("name")
        if not name:
            raise ValueError(f"Tool {tool} is not a valid function")
            # Skip invalid entries; other validators will flag
            continue
        if name in seen:
            return False, f"Duplicate tool detected: name '{name}' appears"
        seen.add(name)
    return True, ""