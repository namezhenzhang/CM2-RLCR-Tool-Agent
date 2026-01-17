import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform checklist JSON: move messages/tools into extra_info and trim messages to first user."
    )
    parser.add_argument(
        "--input_path",
        help="Absolute path to the source JSON file containing a list of items.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional explicit output path. If omitted, writes alongside input with _forverl suffix.",
    )
    return parser.parse_args()


def compute_output_path(input_path: str, explicit_output: Union[str, None]) -> str:
    if explicit_output:
        return explicit_output
    directory, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".json"
    output_filename = f"{name}_all_tools{ext}"
    return os.path.join(directory, output_filename)


def parse_tools_field(tools_value: Any) -> Any:
    # Tools may be a JSON-encoded string or already a Python object.
    if isinstance(tools_value, str):
        return json.loads(tools_value)
    elif isinstance(tools_value, list):
        if tools_value and isinstance(tools_value[0], dict):
            return tools_value
        else:
            return json.loads(tools_value)
    elif isinstance(tools_value, dict):
        return tools_value
    else:
        raise ValueError(f"Unknown tools format: {type(tools_value)}")

def get_called_tools_and_response(item: Dict[str, Any]) -> List[Any]:
    if "extra_info" in item and "messages" in item["extra_info"]:
        messages = item["extra_info"]["messages"]
    else:
        raise ValueError(f"No messages found in item: {item}")
    
    results = defaultdict(list)
    if not isinstance(messages, list):
        raise ValueError(f"Unexpected messages format: {type(messages)}")

    for i in range(len(messages)):
        message = messages[i]
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue

        # Collect contiguous following tool messages for this assistant turn
        following_tool_msgs: List[Dict] = []
        
        for j in range(i+1, i+len(tool_calls)+1):
            assert messages[j].get("role") == "tool" or messages[j].get("role") == "observation", f"Unexpected role: {messages[j].get('role')}"
            following_tool_msgs.append(messages[j]["content"])

        for call, content in zip(tool_calls, following_tool_msgs, strict=True):
            results[call["function"]["name"]].append((json.loads(call["function"]["arguments"]), content))
    return results
    
def get_tools(item: Dict[str, Any]) -> List[Any]:
    if "tools" in item:
        tools = item.get("tools")
    elif "extra_info" in item and "tools" in item["extra_info"]:
        tools = item["extra_info"]["tools"]
    else:
        raise ValueError(f"No tools found in item: {item}")

    if isinstance(tools, str):
        tools = json.loads(tools)
    elif not isinstance(tools, list):
        raise ValueError(f"Unknown tools format: {type(tools)}")
    
    if tools is None:
        return []
    parsed_tools = parse_tools_field(tools)
    if isinstance(parsed_tools, list):
        return parsed_tools
    elif isinstance(parsed_tools, dict):
        return [parsed_tools]
    else:
        raise ValueError(f"Unknown tools format: {type(parsed_tools)}")


def _canonicalize_json_for_compare(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def json_structurally_equal(left_value: Any, right_value: Any) -> bool:
    """
    Return True if two JSON-like Python values are structurally equal.

    This comparison ignores dictionary key ordering by comparing their
    canonical JSON serialization with sorted keys. Lists preserve order
    as in JSON semantics.
    """
    return _canonicalize_json_for_compare(left_value) == _canonicalize_json_for_compare(right_value)


def _coerce_schema_to_dict(schema_obj: Any) -> Dict[str, Any]:
    """
    Convert a schema-like object into a Python dict.
    Accepts raw dict, JSON string, or objects exposing model_dump().
    """
    if isinstance(schema_obj, dict):
        return schema_obj
    if isinstance(schema_obj, str):
        try:
            return json.loads(schema_obj)
        except Exception:
            # Not a JSON string; treat as empty schema
            return {}
    # Try pydantic-like model with model_dump
    dump_fn = getattr(schema_obj, "model_dump", None)
    if callable(dump_fn):
        try:
            return dump_fn()
        except Exception:
            return {}
    return {}


def _json_type_to_python_types(json_type: Any) -> tuple:
    """
    Map JSON Schema type(s) to acceptable Python types.
    Supports single type string or list of type strings.
    """
    mapping = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "object": (dict,),
        "array": (list,),
        "null": (type(None),),
    }
    if isinstance(json_type, list):
        py_types: tuple = tuple()
        for t in json_type:
            py_types += mapping.get(t, (object,))
        return py_types or (object,)
    return mapping.get(json_type, (object,))


def _validate_value_against_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> None:
    # Handle enums first (only when enum is a valid iterable of candidates)
    if isinstance(schema, dict):
        enum_values = schema.get("enum", None)
        if enum_values is not None and isinstance(enum_values, (list, tuple, set)):
            if value not in enum_values:
                raise ValueError(f"{path}: value {value!r} not in enum {enum_values!r}")

    schema_type = schema.get("type") if isinstance(schema, dict) else None

    # If no type given, accept any value (best-effort)
    if not schema_type:
        return

    expected_types = _json_type_to_python_types(schema_type)
    if not isinstance(value, expected_types):
        raise ValueError(f"{path}: expected type {schema_type}, got {type(value).__name__}")

    if schema_type == "object":
        properties: Dict[str, Any] = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required_fields = schema.get("required", []) if isinstance(schema, dict) else []

        # Check required fields
        for key in required_fields:
            if key not in value:
                raise ValueError(f"{path}: missing required property '{key}'")

        # Validate present properties
        for key, val in value.items():
            prop_schema = properties.get(key, {})
            _validate_value_against_schema(val, prop_schema, path=f"{path}.{key}")

    elif schema_type == "array":
        items_schema = schema.get("items", {}) if isinstance(schema, dict) else {}
        for idx, elem in enumerate(value):
            _validate_value_against_schema(elem, items_schema, path=f"{path}[{idx}]")


def validate_arguments(arguments: Dict[str, Any], input_schema: Any) -> None:
    """
    Validate tool arguments against a JSON Schema-like object.

    - Ensures required properties exist for object schemas
    - Performs shallow type checks for primitives, objects, arrays
    - Supports nested validation for objects/arrays and enum constraints
    - Raises ValueError with a concise message on first failure
    """
    schema = _coerce_schema_to_dict(input_schema)

    # If schema explicitly specifies object, ensure arguments is dict
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if schema_type == "object" and not isinstance(arguments, dict):
        raise ValueError(f"$: expected type object, got {type(arguments).__name__}")

    # Validate using recursive helper
    _validate_value_against_schema(arguments, schema, path="$")


def _extract_name_and_params(tool: Any) -> Union[None, tuple]:
    if not isinstance(tool, dict):
        return None
    func = tool.get("function")
    if isinstance(func, dict):
        name = func.get("name")
        params = func.get("parameters")
        return (name, params)
    # Fallback to top-level name/parameters if present
    name = tool.get("name")
    params = tool.get("parameters")
    return (name, params)


def dedupe_and_validate_tools(tools_list: List[Any]) -> List[Any]:
    name_to_params_canon: Dict[str, str] = {}
    name_to_tool: Dict[str, Any] = {}
    seen_full_json: set = set()
    result: List[Any] = []

    for tool in tools_list:
        extracted = _extract_name_and_params(tool)
        if not extracted or extracted[0] is None:
            # No name available: dedupe by full JSON content
            full_key = _canonicalize_json_for_compare(tool)
            if full_key in seen_full_json:
                continue
            seen_full_json.add(full_key)
            result.append(tool)
            continue

        name, params = extracted
        params_canon = _canonicalize_json_for_compare(params)
        if name in name_to_params_canon:
            if name_to_params_canon[name] != params_canon:
                # Conflicting schema for same tool name
                raise ValueError(
                    f"Conflicting tool definitions for name '{name}': parameter schemas differ.\n"
                    f"First: {name_to_params_canon[name]}\nNew:   {params_canon}"
                )
            # Identical schema -> skip duplicate
            continue
        name_to_params_canon[name] = params_canon
        name_to_tool[name] = tool
        result.append(tool)

    return result


def main() -> None:
    args = parse_args()
    input_path = args.input_path
    output_path = compute_output_path(input_path, args.output_path)

    # Read input JSON
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Failed to read JSON from {input_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(
            "Input JSON is expected to be a list of items.",
            file=sys.stderr,
        )
        sys.exit(2)

    all_tools = []
    # Transform all items
    for item in data:
        tools = get_tools(item)
        all_tools.extend(tools)

    # Dedupe and validate
    try:
        all_tools = dedupe_and_validate_tools(all_tools)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(4)


    # Write output JSON
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_tools, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception as exc:
        print(f"Failed to write JSON to {output_path}: {exc}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote transformed file to: {output_path}")


if __name__ == "__main__":
    main()
