import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union


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
    output_filename = f"{name}_forverl{ext}"
    return os.path.join(directory, output_filename)


def find_first_user_message(messages: List[Dict[str, Any]]) -> Union[Dict[str, Any], None]:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def _trim_messages_to_first_user(messages: List[Dict[str, Any]]) -> Union[Dict[str, Any], None]:
    end_index = 0
    for idx, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            end_index = idx
            return messages[:(end_index+1)]
    return []

def transform_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Work on a shallow copy to avoid accidental cross-references
    transformed = dict(item)

    messages = transformed.get("messages")
    tools = transformed.get("tools")
    checklists = transformed.get("checklists")
    extra_info = transformed.get("extra_info") or {}
    uuid = transformed.get("uuid")

    # Move full messages into extra_info
    if messages is not None:
        extra_info["messages"] = messages

        # Trim messages: keep only the first user message + system message if any
        transformed["messages"] = _trim_messages_to_first_user(messages)

    # Convert tools to OpenAI function calling format and save to extra_info.tools
    if tools is not None:
        parsed_tools: List[Dict[str, Any]] = []
        if isinstance(tools, str):
            try:
                parsed = json.loads(tools)
                if isinstance(parsed, list):
                    parsed_tools = parsed
                else:
                    parsed_tools = []
            except Exception:
                parsed_tools = []
        elif isinstance(tools, list):
            parsed_tools = tools
        else:
            parsed_tools = []

        # Convert to OpenAI function calling format if needed
        # OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        # Simple format: {"name": ..., "description": ..., "parameters": ...}
        openai_format_tools: List[Dict[str, Any]] = []
        for tool in parsed_tools:
            if isinstance(tool, dict):
                # Check if already in OpenAI format
                if tool.get("type") == "function" and "function" in tool:
                    openai_format_tools.append(tool)
                # Convert simple format to OpenAI format
                elif "name" in tool:
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    }
                    openai_format_tools.append(openai_tool)

        # Save the OpenAI format tool schemas for tokenizer usage
        if openai_format_tools:
            extra_info["tools"] = json.dumps(openai_format_tools)

            # Build minimal tools_kwargs: map each tool name to an empty config
            # tool_kwargs: Dict[str, Any] = {}
            # for tool in parsed_tools:
            #     try:
            #         if isinstance(tool, dict) and tool.get("type") == "function":
            #             fn = tool.get("function", {})
            #             name = fn.get("name")
            #             if isinstance(name, str) and name:
            #                 # Create stub entry; users/configs can enrich create/execute kwargs downstream
            #                 tool_kwargs[name] = tool_kwargs.get(name, {"create_kwargs": {"_": 1}, "execute_kwargs": {"_": 1},})
            #     except Exception:
            #         # Robust to malformed tool entries
            #         continue

            # if tool_kwargs:
            #     extra_info.setdefault("tools_kwargs", {})
            #     # Merge but do not overwrite any existing keys
            #     for k, v in tool_kwargs.items():
            #         if k not in extra_info["tools_kwargs"]:
            #             extra_info["tools_kwargs"][k] = v

        # Remove top-level tools field after migrating to extra_info
        if "tools" in transformed:
            del transformed["tools"]

    # Move checklist into extra_info.interaction_kwargs
    if checklists is not None:
        extra_info["interaction_kwargs"] = {
            "name": "checklist",
            "checklist_list": checklists,
        }
        # Remove top-level checklist
        if "checklist" in transformed:
            del transformed["checklist"]

    extra_info["need_tools_kwargs"] = True
    extra_info["original_index"] = uuid
    transformed["extra_info"] = extra_info
    return transformed


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

    # Transform all items
    transformed_items = [transform_item(item) for item in data]

    # Write output JSON
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transformed_items, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception as exc:
        print(f"Failed to write JSON to {output_path}: {exc}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote transformed file to: {output_path}")


if __name__ == "__main__":
    main()
