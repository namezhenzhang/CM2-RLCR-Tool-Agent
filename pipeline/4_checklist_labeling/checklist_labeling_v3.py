from openai import OpenAI
import os
import pathlib
import json
import re
import sys
from tqdm import tqdm
import argparse
from datasets import load_from_disk
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import signal
import uuid as generate_uuid
# ANSI color codes for readable logs
CLR_RESET = "\033[0m"
CLR_BLUE = "\033[34m"       # info
CLR_YELLOW = "\033[33m"     # warning
CLR_RED = "\033[31m"        # error
CLR_GREEN = "\033[32m"      # success
CLR_MAGENTA = "\033[35m"    # save events
CLR_CYAN = "\033[36m"       # price/usage

def _color(text: str, color: str) -> str:
    return f"{color}{text}{CLR_RESET}"

# Model pricing configuration (USD per 1M tokens)
MODEL_PRICING = {
    "gpt-5": {
        "input": 1.250,
        "cached_input": 0.125,
        "output": 10.000
    },
    "gpt-5-mini": {
        "input": 0.250,
        "cached_input": 0.025,
        "output": 2.000
    },
    "gpt-5-nano": {
        "input": 0.050,
        "cached_input": 0.005,
        "output": 0.400
    }
}


# Strict JSON schema for response_format to eliminate tag-based parsing
CHECKLIST_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "turn": {"type": "integer", "minimum": 0},
            "checklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "turn": {"type": "integer", "minimum": 0},
                                    "step": {"type": "integer", "minimum": 0},
                                    "from": {
                                        "type": "string",
                                        "enum": [
                                            "user.content",
                                            "assistant.content.thinking",
                                            "assistant.content.user_visible_reply",
                                            "assistant.tool_calls",
                                            "tool.content",
                                        ],
                                    },
                                    "snippet": {"type": "string"},
                                },
                                "required": ["turn", "step", "from", "snippet"],
                            },
                        },
                        "focus_on": {
                            "type": "string",
                            "enum": [
                                "assistant.tool_calls",
                                "assistant.content.thinking",
                                "assistant.content.user_visible_reply",
                                "tool.content"
                            ]
                        },
                        "question": {"type": "string"},
                        "pass_condition": {"type": "string"},
                        "failure_examples": {"type": "array", "items": {"type": "string"}},
                        "required_for_next_turn": {"type": "boolean"}
                    },
                    "required": [
                        "id",
                        "evidence",
                        "focus_on",
                        "question",
                        "pass_condition",
                        "failure_examples",
                        "required_for_next_turn"
                    ],
                },
            },
            "dependence": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "weight": {"type": "object", "additionalProperties": {"type": "number"}},
        },
        "required": ["turn", "checklist", "dependence", "weight"],
    },
}

def get_model_pricing(model_name: str) -> dict:
    """Get pricing configuration for a model, with fallbacks for similar names"""
    # Direct match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    
    # Fallback matching for common variations
    model_lower = model_name.lower().replace("_", "-").replace(" ", "-")
    for key in MODEL_PRICING:
        if key.lower().replace("_", "-").replace(" ", "-") == model_lower:
            return MODEL_PRICING[key]
    
    # Check for partial matches (e.g., "gpt5" -> "gpt-5")
    for key in MODEL_PRICING:
        key_clean = key.lower().replace("-", "").replace("_", "").replace(" ", "")
        model_clean = model_lower.replace("-", "").replace("_", "").replace(" ", "")
        if key_clean == model_clean:
            return MODEL_PRICING[key]
    
    # Default to GPT-5 pricing if no match found
    print(_color(f"Warning: No pricing found for model '{model_name}', using GPT-5 pricing as default", CLR_YELLOW), file=sys.stderr)
    return MODEL_PRICING["gpt-5"]

def _validate_checklist_json(data: list[dict]):
    if not isinstance(data, list):
        raise ValueError("Top-level must be an array")
    # Accept both new fully-qualified names and legacy short names for backward compatibility
    allowed_from = {
        # New schema values
        "user.content",
        "assistant.content.thinking",
        "assistant.content.user_visible_reply",
        "assistant.tool_calls",
        "tool.content"
    }
    allowed_focus_on = {
        "assistant.tool_calls",
        "assistant.content.thinking",
        "assistant.content.user_visible_reply",
        "tool.content",
    }
    for i, block in enumerate(data):
        if not isinstance(block, dict):
            raise ValueError(f"Item {i} must be an object")
        for k in ("turn", "checklist", "dependence", "weight"):
            if k not in block:
                raise ValueError(f"Item {i} missing required key: {k}")
        if not (isinstance(block["turn"], int) and block["turn"] >= 0):
            raise ValueError(f"Item {i}.turn must be non-negative integer")

        checklist = block["checklist"]
        if not isinstance(checklist, list):
            raise ValueError(f"Item {i}.checklist must be an array")
        for j, ci in enumerate(checklist):
            if not isinstance(ci, dict):
                raise ValueError(f"Item {i}.checklist[{j}] must be an object")
            for req in ("id", "evidence", "focus_on", "question", "pass_condition", "failure_examples", "required_for_next_turn"):
                if req not in ci:
                    raise ValueError(f"Item {i}.checklist[{j}] missing required key: {req}")
            if not isinstance(ci["id"], str) or not ci["id"]:
                raise ValueError(f"Item {i}.checklist[{j}].id must be non-empty string")
            if not isinstance(ci["evidence"], list):
                raise ValueError(f"Item {i}.checklist[{j}].evidence must be an array")
            for k, ev in enumerate(ci["evidence"]):
                if not isinstance(ev, dict):
                    raise ValueError(f"Item {i}.checklist[{j}].evidence[{k}] must be object")
                for req_ev in ("turn", "step", "from", "snippet"):
                    if req_ev not in ev:
                        raise ValueError(f"Item {i}.checklist[{j}].evidence[{k}] missing key: {req_ev}")
                if not (isinstance(ev["turn"], int) and ev["turn"] >= 0):
                    raise ValueError(f"Item {i}.checklist[{j}].evidence[{k}].turn must be non-negative integer")
                if not (isinstance(ev["step"], int) and ev["step"] >= 0):
                    raise ValueError(f"Item {i}.checklist[{j}].evidence[{k}].step must be non-negative integer")
                if ev["from"] not in allowed_from:
                    raise ValueError(f"Item {i}.checklist[{j}].evidence[{k}].from invalid: {ev['from']}")
                if not isinstance(ev["snippet"], str):
                    raise ValueError(f"Item {i}.checklist[{j}].evidence[{k}].snippet must be string")
            if not isinstance(ci["focus_on"], str) or ci["focus_on"] not in allowed_focus_on:
                raise ValueError(f"Item {i}.checklist[{j}].focus_on invalid: {ci['focus_on']}")
            if not isinstance(ci["question"], str):
                raise ValueError(f"Item {i}.checklist[{j}].question must be string")
            if not isinstance(ci["pass_condition"], str):
                raise ValueError(f"Item {i}.checklist[{j}].pass_condition must be string")
            if not (isinstance(ci["failure_examples"], list) and all(isinstance(x, str) for x in ci["failure_examples"])):
                raise ValueError(f"Item {i}.checklist[{j}].failure_examples must be array of strings")
            if not isinstance(ci["required_for_next_turn"], bool):
                raise ValueError(f"Item {i}.checklist[{j}].required_for_next_turn must be boolean")


        dep = block["dependence"]
        if not isinstance(dep, dict):
            raise ValueError(f"Item {i}.dependence must be object")
        for key, val in dep.items():
            if not isinstance(key, str):
                raise ValueError(f"Item {i}.dependence key must be string")
            if not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
                raise ValueError(f"Item {i}.dependence['{key}'] must be array of strings")

        # Enforce dependence keys exactly match checklist ids (no missing/extra)
        id_set = {ci["id"] for ci in checklist}
        dep_keys = set(dep.keys())
        if dep_keys != id_set:
            extra = sorted(list(dep_keys - id_set))
            missing = sorted(list(id_set - dep_keys))
            details = []
            if extra:
                details.append(f"extra keys: {extra}")
            if missing:
                details.append(f"missing keys: {missing}")
            raise ValueError(
                f"Item {i}.dependence keys must match checklist ids; " + "; ".join(details)
            )

        weight = block["weight"]
        if not isinstance(weight, dict):
            raise ValueError(f"Item {i}.weight must be object")
        for key, val in weight.items():
            if not isinstance(key, str):
                raise ValueError(f"Item {i}.weight key must be string")
            if not (isinstance(val, int) or isinstance(val, float)):
                raise ValueError(f"Item {i}.weight['{key}'] must be number")
        
        # Validate that weights sum to 1.0
        weight_sum = sum(weight.values())
        if abs(weight_sum - 1.0) > 1e-6:  # Allow small floating point errors
            raise ValueError(f"Item {i}.weight values must sum to 1.0, but sum is {weight_sum}")

        # Enforce: referenced dependencies must focus on 'tool.content'
        id_to_focus_on = {ci["id"]: ci["focus_on"] for ci in checklist}
        for dependent_id, referenced_ids in dep.items():
            for referenced_id in referenced_ids:
                if referenced_id not in id_to_focus_on:
                    raise ValueError(
                        f"Item {i}.dependence['{dependent_id}'] references unknown id: {referenced_id}"
                    )
                if id_to_focus_on[referenced_id] != "tool.content":
                    raise ValueError(
                        f"Item {i}.dependence['{dependent_id}'] references '{referenced_id}' which must have focus_on 'tool.content'"
                    )

def _validate_existing_sample(sample: dict) -> tuple[bool, str, int]:
    """Validate a sample that may hold a single checklist or a list of checklist variants.

    Returns (ok, reason, count) where count is the number of valid checklist variants present.
    """
    try:
        # If multiple variants exist
        if isinstance(sample.get("checklists"), list):
            variants = sample.get("checklists")
            for variant_idx, checklist in enumerate(variants):
                if not isinstance(checklist, list):
                    return False, f"checklists[{variant_idx}] is not a list", 0
                for turn_idx, turn_checklist in enumerate(checklist):
                    if not isinstance(turn_checklist, list):
                        return False, f"checklists[{variant_idx}][turn={turn_idx}] is not a list", 0
                    turn_block = {"turn": turn_idx, "checklist": [], "dependence": {}, "weight": {}}
                    for item in turn_checklist:
                        if not isinstance(item, dict):
                            return False, f"checklists[{variant_idx}][turn={turn_idx}] item is not an object", 0
                        item_id = item.get("id")
                        if not isinstance(item_id, str) or not item_id:
                            return False, f"checklists[{variant_idx}][turn={turn_idx}] invalid id", 0
                        checklist_item = {
                            "id": item_id,
                            "evidence": item.get("evidence"),
                            "focus_on": item.get("focus_on"),
                            "question": item.get("question"),
                            "pass_condition": item.get("pass_condition"),
                            "failure_examples": item.get("failure_examples"),
                            "required_for_next_turn": item.get("required_for_next_turn"),
                        }
                        turn_block["checklist"].append(checklist_item)
                        dependence = item.get("dependence")
                        weight = item.get("weight")
                        if not isinstance(dependence, list):
                            return False, f"checklists[{variant_idx}] dependence for id '{item_id}' is not a list", 0
                        if not (isinstance(weight, (int, float))):
                            return False, f"checklists[{variant_idx}] weight for id '{item_id}' is not a number", 0
                        turn_block["dependence"][item_id] = dependence
                        turn_block["weight"][item_id] = weight
                    _validate_checklist_json([turn_block])
            return True, "", len(variants)

        # Otherwise expect a single checklist at key 'checklist'
        checklist = sample.get("checklist")
        if not isinstance(checklist, list):
            return False, "checklist is not a list", 0
        for turn_idx, turn_checklist in enumerate(checklist):
            if not isinstance(turn_checklist, list):
                return False, f"turn {turn_idx} checklist is not a list", 0
            turn_block = {"turn": turn_idx, "checklist": [], "dependence": {}, "weight": {}}
            for item in turn_checklist:
                if not isinstance(item, dict):
                    return False, f"turn {turn_idx} checklist item is not an object", 0
                item_id = item.get("id")
                if not isinstance(item_id, str) or not item_id:
                    return False, f"turn {turn_idx} checklist item has invalid id", 0
                checklist_item = {
                    "id": item_id,
                    "evidence": item.get("evidence"),
                    "focus_on": item.get("focus_on"),
                    "question": item.get("question"),
                    "pass_condition": item.get("pass_condition"),
                    "failure_examples": item.get("failure_examples"),
                    "required_for_next_turn": item.get("required_for_next_turn"),
                }
                turn_block["checklist"].append(checklist_item)
                dependence = item.get("dependence")
                weight = item.get("weight")
                if not isinstance(dependence, list):
                    return False, f"dependence for id '{item_id}' is not a list", 0
                if not (isinstance(weight, (int, float))):
                    return False, f"weight for id '{item_id}' is not a number", 0
                turn_block["dependence"][item_id] = dependence
                turn_block["weight"][item_id] = weight
            _validate_checklist_json([turn_block])
        return True, "", 1
    except Exception as e:
        return False, str(e), 0

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        if len(parts) >= 2:
            s = parts[1]
            s_lines = s.splitlines()
            if s_lines:
                first_line = s_lines[0].strip()
                if first_line and not first_line.startswith("[") and not first_line.startswith("{"):
                    s_lines = s_lines[1:]
            s = "\n".join(s_lines)
        if "```" in s:
            s = s.rsplit("```", 1)[0]
        return s.strip()
    return s

def _extract_json_string(text: str) -> str:
    """Extract a JSON string from free-form model text.

    Tries raw, stripped code fences, bracket slicing for arrays, and fenced blocks.
    """
    # Try raw
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Try stripped fences
    stripped = _strip_code_fences(text)
    if stripped != text:
        try:
            json.loads(stripped)
            return stripped
        except Exception:
            pass

    # Try bracket slice (top-level array expected)
    source = stripped if stripped is not None else text
    try:
        start = source.find("[")
        end = source.rfind("]") + 1
        if start != -1 and end > start:
            candidate = source[start:end]
            json.loads(candidate)
            return candidate
    except Exception:
        pass

    # Try each fenced block
    try:
        for m in re.finditer(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, re.DOTALL):
            block = m.group(1).strip()
            try:
                json.loads(block)
                return block
            except Exception:
                continue
    except Exception:
        pass

    return text


def _parse_result_json(text: str) -> list[dict]:
    """Parse model output: extract JSON and validate against the expected schema."""
    json_text = _extract_json_string(text)
    data = json.loads(json_text)
    _validate_checklist_json(data)
    data_sorted = sorted(data, key=lambda b: int(b.get("turn", 0)))
    return data_sorted


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="../data/llm_based_filtered_data/llm_based_filtered_high4.json")
parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
parser.add_argument("--num_checklists", type=int, default=1, help="Generate this many checklist variants per sample")
parser.add_argument("--output_file", type=str, default="../data/rl_data/checklist_annotated.json")
parser.add_argument("--prompt_file", type=str, default="checklist_prompt_v3.txt", help="Prompt file")

parser.add_argument("--model", type=str, default="gpt-5", help="Model name (pricing will be auto-detected)")
parser.add_argument("--price-input-per-m", type=float, default=None, help="Override input price (USD per 1M tokens)")
parser.add_argument(
    "--price-cached-input-per-m", type=float, default=None, help="Override cached input price (USD per 1M tokens)"
)
parser.add_argument("--price-output-per-m", type=float, default=None, help="Override output price (USD per 1M tokens)")

parser.add_argument("--effort", type=str, default="medium", choices=["minimal", "low", "medium", "high"], help="Effort level for reasoning")
parser.add_argument("--verbosity", type=str, default="medium", choices=["low", "medium", "high"], help="Verbosity level for output")
parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("--retry", type=int, default=3)
parser.add_argument("--save_every", type=int, default=5, help="Save partial results every N items")
parser.add_argument("--save_every_seconds", type=int, default=60, help="Save partial results every N seconds")
parser.add_argument("--enable-cache-control", action="store_true", help="Use structured system content; caching is automatic when prefix is identical")
parser.add_argument("--cache-warmup", action="store_true", help="Send a light warmup call to seed prompt cache")
parser.add_argument("--warmup-max-output-tokens", type=int, default=16, help="Max output tokens for warmup call (min 16)")
args = parser.parse_args()

# Set up client
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key == "":
    raise ValueError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

# Load data
current_folder = pathlib.Path(__file__).parent
with open(args.prompt_file, "r") as f:
    checklist_prompt = f.read()

# with open(current_folder / "../filtered_data/nvidia_nemotron_v1_tool_calling_qwen3.json", "r") as f:
#     data = json.load(f)

dataset_dir = args.dataset_dir
if dataset_dir.endswith(".json"):
    with open(dataset_dir,"r") as f:
        ds = json.load(f)
else:
    ds = load_from_disk(dataset_dir)

if args.n_samples and args.n_samples > 0 and args.n_samples < len(ds):
    rng = np.random.RandomState(seed=args.seed)
    indices = rng.choice(len(ds), args.n_samples, replace=False).tolist()
else:
    indices = list(range(len(ds)))

data = []
for idx in indices:
    sample = ds[int(idx)]
    uuid = sample.get("uuid", str(generate_uuid.uuid4()))
    messages = sample.get("messages", [])
    if dataset_dir.endswith(".json"):
        tools = sample.get("tools", [])
    else:
        metadata = json.loads(sample.get("metadata", "{}"))
        tools = metadata.get("tools", [])
    data.append({
        "uuid": uuid,
        "messages": messages,
        "tools": tools,
    })

"""
Resume support: if an existing annotated file is present, load it and skip
any items with matching UUIDs. Pre-seed result_list with existing results
so partial and final saves include them.
"""
existing_results: list[dict] = []
existing_uuids: set[str] = set()
try:
    existing_path = current_folder / "checklist_data/checklist_annotated.json"
    if existing_path.exists():
        with open(existing_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                valid_samples = []
                invalid_count = 0
                
                for _it in loaded:
                    try:
                        _u = _it.get("uuid")
                        if isinstance(_u, str):
                            # Validate the sample structure (single or multi variant)
                            _is_valid, _reason, _count = _validate_existing_sample(_it)
                            if _is_valid:
                                # Normalize to 'checklists' list
                                if "checklists" not in _it:
                                    if "checklist" in _it:
                                        _it["checklists"] = [_it["checklist"]]
                                        _it.pop("checklist", None)
                                    else:
                                        _it["checklists"] = []
                                valid_samples.append(_it)
                                existing_uuids.add(_u)
                            else:
                                invalid_count += 1
                                print(_color(f"[resume] Filtered out invalid sample UUID={_u}: {_reason}", CLR_YELLOW), file=sys.stderr, flush=True)
                    except Exception as e:
                        invalid_count += 1
                        print(_color(f"[resume] Failed to process sample UUID={_it.get('uuid')}: {e}", CLR_YELLOW), file=sys.stderr, flush=True)
                        continue
                
                existing_results = valid_samples
                if invalid_count > 0:
                    print(_color(f"[resume] Filtered out {invalid_count} invalid samples from existing results", CLR_YELLOW), file=sys.stderr, flush=True)
except Exception as e:
    print(_color(f"[resume] Failed to load existing results: {e}", CLR_YELLOW), file=sys.stderr, flush=True)

if existing_uuids:
    original_total = len(data)
    # Do not skip items entirely when multi-variants are requested; allow topping up variants
    # We keep all items and rely on the scheduling logic to add only missing variants.
    data = data
    skipped = original_total - len(data)
    print(_color(f"Resuming: existing {skipped} already present; planning to top up to {args.num_checklists} variants per sample", CLR_BLUE), flush=True)
else:
    print(_color(f"Annotating {len(data)} samples", CLR_BLUE), flush=True)

"""
Accumulate previous usage & cost if a usage summary exists.
This allows usage.json to reflect totals across resumed runs.
"""
base_requests = 0
base_input_tokens = 0
base_cached_input_tokens = 0
base_output_tokens = 0
base_estimated_cost_usd = 0.0
try:
    prev_usage_path = current_folder / "checklist_data/checklist_annotated.usage.json"
    if prev_usage_path.exists():
        with open(prev_usage_path, "r") as f:
            prev_usage = json.load(f)
            base_requests = int(prev_usage.get("requests", 0) or 0)
            base_input_tokens = int(prev_usage.get("input_tokens", 0) or 0)
            base_cached_input_tokens = int(prev_usage.get("cached_input_tokens", 0) or 0)
            base_output_tokens = int(prev_usage.get("output_tokens", 0) or 0)
            base_estimated_cost_usd = float(prev_usage.get("estimated_cost_usd", 0) or 0.0)
        print(
            _color(
                (
                    f"[resume_usage] previous: requests={base_requests} input={base_input_tokens} "
                    f"(cached={base_cached_input_tokens}) output={base_output_tokens} "
                    f"cost=${base_estimated_cost_usd:.6f}"
                ),
                CLR_CYAN,
            ),
            file=sys.stderr,
            flush=True,
        )
except Exception as e:  # noqa: BLE001
    print(f"[resume_usage] Failed to load previous usage: {e}", file=sys.stderr, flush=True)



# ---- Usage & Cost tracking helpers ----

# Get model pricing (auto-detect from model name or use CLI overrides)
model_pricing = get_model_pricing(args.model)
_PRICE_PER_INPUT_TOKEN = (args.price_input_per_m or model_pricing["input"]) / 1_000_000
_PRICE_PER_CACHED_INPUT_TOKEN = (args.price_cached_input_per_m or model_pricing["cached_input"]) / 1_000_000
_PRICE_PER_OUTPUT_TOKEN = (args.price_output_per_m or model_pricing["output"]) / 1_000_000

# Print pricing information
print(_color(f"Using model: {args.model}", CLR_BLUE), file=sys.stderr)
print(_color(f"Pricing (per 1M tokens): input=${model_pricing['input']:.3f}, cached=${model_pricing['cached_input']:.3f}, output=${model_pricing['output']:.3f}", CLR_CYAN), file=sys.stderr)
if args.price_input_per_m or args.price_cached_input_per_m or args.price_output_per_m:
    print(_color("Note: Some prices overridden via command line arguments", CLR_YELLOW), file=sys.stderr)
print("", file=sys.stderr)  # Empty line for better readability

def _getattr_or_key(obj, key: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def _extract_usage_dict(resp) -> dict:
    usage = _getattr_or_key(resp, "usage", None)
    input_tokens = int(_getattr_or_key(usage, "input_tokens", 0) or 0)
    output_tokens = int(_getattr_or_key(usage, "output_tokens", 0) or 0)
    # Cached tokens (if present)
    input_token_details = _getattr_or_key(usage, "input_token_details", None)
    cached_input_tokens = int(_getattr_or_key(input_token_details, "cached_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
    }

def _estimate_cost_usd(input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    non_cached_input = max(0, input_tokens - cached_input_tokens)
    return (
        non_cached_input * _PRICE_PER_INPUT_TOKEN
        + cached_input_tokens * _PRICE_PER_CACHED_INPUT_TOKEN
        + output_tokens * _PRICE_PER_OUTPUT_TOKEN
    )

result_list = list(existing_results)
total_requests = 0
total_input_tokens = 0
total_cached_input_tokens = 0
total_output_tokens = 0
total_cost_usd = 0.0

_agg_lock = threading.Lock()
progress_completed = 0
PROGRESS_TOTAL = 0

def _atomic_json_dump(path: pathlib.Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)

def _save_checkpoint(checkpoint: bool, progress: int):
    """Save partial or final results and usage with colored logs."""
    # Save results
    _atomic_json_dump(
        current_folder / "checklist_data/checklist_annotated.json",
        result_list,
    )
    print(_color("[save] Wrote partial checklist_annotated.json" if checkpoint else "[save] Wrote final checklist_annotated.json", CLR_MAGENTA), file=sys.stderr, flush=True)

    # Compose usage
    combined_requests = base_requests + total_requests
    combined_input_tokens = base_input_tokens + total_input_tokens
    combined_cached_input_tokens = base_cached_input_tokens + total_cached_input_tokens
    combined_output_tokens = base_output_tokens + total_output_tokens
    combined_non_cached_input = max(0, combined_input_tokens - combined_cached_input_tokens)
    combined_cost_usd = round(base_estimated_cost_usd + _estimate_cost_usd(total_input_tokens, total_cached_input_tokens, total_output_tokens), 6)
    usage_summary_payload = {
        "model": args.model,
        "requests": combined_requests,
        "input_tokens": combined_input_tokens,
        "cached_input_tokens": combined_cached_input_tokens,
        "non_cached_input_tokens": combined_non_cached_input,
        "output_tokens": combined_output_tokens,
        "estimated_cost_usd": combined_cost_usd,
        "pricing_per_million": {
            "input": args.price_input_per_m or model_pricing["input"],
            "cached_input": args.price_cached_input_per_m or model_pricing["cached_input"],
            "output": args.price_output_per_m or model_pricing["output"],
        },
        "pricing_source": "auto_detected" if not any([args.price_input_per_m, args.price_cached_input_per_m, args.price_output_per_m]) else "mixed",
    }
    # Enrich with progress when checkpointing
    if checkpoint:
        usage_summary_payload.update({
            "progress_completed": progress,
            "progress_total": PROGRESS_TOTAL,
            "checkpoint": True,
        })
    _atomic_json_dump(
        current_folder / "checklist_data/checklist_annotated.usage.json",
        usage_summary_payload,
    )
    print(_color((f"[save] Wrote partial usage summary. est_cost=${combined_cost_usd:.6f}" if checkpoint else f"[save] Wrote final usage summary. est_cost=${combined_cost_usd:.6f}"), CLR_MAGENTA), file=sys.stderr, flush=True)

def _handle_sigterm(signum, frame):
    try:
        print(_color(f"[signal] Received SIGTERM ({signum}). Saving checkpoint...", CLR_RED), file=sys.stderr, flush=True)
        with _agg_lock:
            _save_checkpoint(True, progress_completed)
    finally:
        try:
            client.close()
        except Exception:
            pass
        sys.exit(143)

def _build_system_content():
    """Build system content, optionally marking it as cacheable.

    When --enable-cache-control is set, wrap the system text as a content part
    with cache_control to encourage prefix caching.
    """
    if args.enable_cache_control:
        return [
            {
                "type": "input_text",
                "text": checklist_prompt,
            }
        ]
    return checklist_prompt

def _process_one(i: int, item: dict) -> dict:
    local_requests = 0
    local_input_tokens = 0
    local_cached_input_tokens = 0
    local_output_tokens = 0
    attempt_cost_usd = 0.0
    last_error: Exception | None = None

    # Reuse the global client to avoid creating many HTTP clients (prevents FD leaks)

    tools = item.get("tools", [])
    messages = item.get("messages", [])
    messages_and_tools = f"# Candidate Tools: {json.dumps(tools, ensure_ascii=False, indent=4)}\n# Messages: {json.dumps(messages, ensure_ascii=False, indent=4)}"
    inputs = [
        {"role": "system", "content": _build_system_content()},
        {"role": "user", "content": messages_and_tools},
    ]

    for attempt in range(args.retry):
        result = client.responses.create(
            model=args.model,
            input=inputs,
            reasoning={"effort": args.effort},  # minimal, low, medium, and high
            text={"verbosity": args.verbosity},  # low, medium, and high
        )
        # Track usage and cost for every API call (including retries)
        usage = _extract_usage_dict(result)
        local_requests += 1
        local_input_tokens += usage["input_tokens"]
        local_cached_input_tokens += usage["cached_input_tokens"]
        local_output_tokens += usage["output_tokens"]
        call_cost = _estimate_cost_usd(
            usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"]
        )
        attempt_cost_usd += call_cost
        print(
            _color(
                f"[usage] i={i} attempt={attempt + 1}: input={usage['input_tokens']} (cached={usage['cached_input_tokens']}) "
                f"output={usage['output_tokens']} cost=${call_cost:.6f}",
                CLR_CYAN,
            ),
            file=sys.stderr,
            flush=True,
        )

        output_text = result.output_text
        # print(output_text)
        try:
            parsed_blocks = _parse_result_json(output_text)
            expected_user_turns = 0
            try:
                original_prompt = messages
                if isinstance(original_prompt, list):
                    expected_user_turns = sum(
                        1
                        for msg in original_prompt
                        if isinstance(msg, dict) and msg.get("role") == "user"
                    )
            except Exception:
                expected_user_turns = 0
            if len(parsed_blocks) != expected_user_turns:
                raise ValueError(
                    f"turn count mismatch: parsed={len(parsed_blocks)} vs user_turns={expected_user_turns}"
                )

            final_per_turn: list[list[dict]] = []
            for block in parsed_blocks:
                checklist_items: list[dict] = block["checklist"]
                if isinstance(checklist_items, list) and len(checklist_items) > 0:
                    enriched_items: list[dict] = []
                    dependence_map = block.get("dependence", {})
                    weight_map = block.get("weight", {})
                    for idx_in_turn, ci in enumerate(checklist_items):
                        new_item = dict(ci)
                        item_id = new_item.get("id") or f"C{idx_in_turn}"
                        if item_id in dependence_map:
                            new_item["dependence"] = dependence_map[item_id]
                        else:
                            raise ValueError(f"Item {item_id} not found in dependence map")

                        if item_id in weight_map:
                            new_item["weight"] = weight_map[item_id]
                        else:
                            raise ValueError(f"Item {item_id} not found in weight map")

                        extra_info = dict(new_item.get("extra_info", {}))
                        extra_info["price_usd"] = round(attempt_cost_usd, 6)
                        new_item["extra_info"] = extra_info
                        enriched_items.append(new_item)
                    checklist_items = enriched_items
                final_per_turn.append(checklist_items)

            print(_color(f"Turn {i} done, cost=${attempt_cost_usd:.6f}", CLR_GREEN), flush=True)
            new_data = item.copy()
            new_data["uuid"] = item.get("uuid")
            # Return a single generated checklist variant
            new_data["checklist"] = final_per_turn
            return {
                "ok": True,
                "i": i,
                "attempt_cost_usd": attempt_cost_usd,
                "requests": local_requests,
                "input_tokens": local_input_tokens,
                "cached_input_tokens": local_cached_input_tokens,
                "output_tokens": local_output_tokens,
                "data": new_data,
            }
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(
                _color(f"[checklist_parser] Parse failed (i={i}, attempt={attempt + 1}): {e}", CLR_RED),
                file=sys.stderr,
                flush=True,
            )
            # continue to retry

    return {
        "ok": False,
        "i": i,
        "reason": str(last_error) if last_error else "unknown",
        "requests": local_requests,
        "input_tokens": local_input_tokens,
        "cached_input_tokens": local_cached_input_tokens,
        "output_tokens": local_output_tokens,
    }


# Optional cache warmup before parallel processing
if args.cache_warmup:
    try:
        print(_color("[cache_warmup] Seeding prompt cache...", CLR_BLUE), file=sys.stderr, flush=True)
        warmup_inputs = [
            {"role": "system", "content": _build_system_content()},
            {"role": "user", "content": "warmup"},
        ]
        warmup_resp = client.responses.create(
            model=args.model,
            input=warmup_inputs,
            reasoning={"effort": args.effort},
            text={"verbosity": args.verbosity},
            max_output_tokens=max(16, int(args.warmup_max_output_tokens or 16)),
        )
        warmup_usage = _extract_usage_dict(warmup_resp)
        print(
            _color(
                f"[cache_warmup] input={warmup_usage['input_tokens']} (cached={warmup_usage['cached_input_tokens']}) output={warmup_usage['output_tokens']}",
                CLR_CYAN,
            ),
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(_color(f"[cache_warmup] Failed: {e}", CLR_RED), file=sys.stderr, flush=True)

signal.signal(signal.SIGTERM, _handle_sigterm)

try:
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        # Build tasks considering desired number of checklist variants per sample
        tasks: list[tuple[int, dict]] = []
        for i in range(len(data)):
            item = data[i]
            uuid = item.get("uuid")
            # Find if this uuid already has results; compute how many more to generate
            existing_variants = 0
            for res in existing_results:
                if res.get("uuid") == uuid:
                    if isinstance(res.get("checklists"), list):
                        existing_variants = max(existing_variants, len(res.get("checklists")))
                    elif isinstance(res.get("checklist"), list):
                        existing_variants = max(existing_variants, 1)
            needed = max(0, int(args.num_checklists) - int(existing_variants))
            if needed == 0 and uuid in existing_uuids:
                continue
            # Schedule 'needed' copies; each call generates one variant
            for _ in range(max(1, needed)):
                tasks.append((i, item))
        PROGRESS_TOTAL = len(tasks)
        futures = [executor.submit(_process_one, i, item) for (i, item) in tasks]
        completed = 0
        last_save_time = time.time()
        with tqdm(total=PROGRESS_TOTAL) as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                with _agg_lock:
                    total_requests += res.get("requests", 0)
                    total_input_tokens += res.get("input_tokens", 0)
                    total_cached_input_tokens += res.get("cached_input_tokens", 0)
                    total_output_tokens += res.get("output_tokens", 0)
                    # Recompute cost from tokens to avoid floating point accumulation from threads
                    total_cost_usd = _estimate_cost_usd(total_input_tokens, total_cached_input_tokens, total_output_tokens)

                    if res.get("ok"):
                        generated = res["data"]
                        gen_uuid = generated.get("uuid")
                        # Merge into result_list: normalize to checklists[]
                        merged = False
                        for idx_r, r in enumerate(result_list):
                            if r.get("uuid") == gen_uuid:
                                if "checklists" not in r:
                                    if "checklist" in r and isinstance(r.get("checklist"), list):
                                        r["checklists"] = [r["checklist"]]
                                        r.pop("checklist", None)
                                    else:
                                        r["checklists"] = []
                                r["checklists"].append(generated.get("checklist"))
                                merged = True
                                break
                        if not merged:
                            # New entry
                            out = {k: v for k, v in generated.items() if k not in {"checklists"}}
                            out["checklists"] = [generated.get("checklist")]
                            out.pop("checklist", None)
                            result_list.append(out)
                    else:
                        err_payload = {
                            "error": "parse_failed",
                            "index": res.get("i"),
                            "reason": res.get("reason", "unknown"),
                        }
                        print(json.dumps(err_payload, ensure_ascii=False), flush=True)

                    completed += 1
                    progress_completed = completed
                    now = time.time()
                    should_save_count = args.save_every > 0 and (completed % args.save_every == 0)
                    should_save_time = args.save_every_seconds > 0 and (now - last_save_time >= args.save_every_seconds)
                    if should_save_count or should_save_time:
                        _save_checkpoint(True, completed)
                        last_save_time = now
                    pbar.update(1)
except KeyboardInterrupt:
    print(_color("[signal] KeyboardInterrupt received. Saving checkpoint and exiting...", CLR_RED), file=sys.stderr, flush=True)
    with _agg_lock:
        _save_checkpoint(True, progress_completed)
    try:
        client.close()
    except Exception:
        pass
    sys.exit(130)

_atomic_json_dump(pathlib.Path(args.output_file), result_list)
print(_color(f"[save] Wrote final {args.output_file}", CLR_MAGENTA), file=sys.stderr, flush=True)

# Final usage summary (stderr) and JSON dump alongside output
combined_requests = base_requests + total_requests
combined_input_tokens = base_input_tokens + total_input_tokens
combined_cached_input_tokens = base_cached_input_tokens + total_cached_input_tokens
combined_output_tokens = base_output_tokens + total_output_tokens
combined_non_cached_input = max(0, combined_input_tokens - combined_cached_input_tokens)
combined_cost_usd = round(base_estimated_cost_usd + total_cost_usd, 6)
print(
    (
        f"[usage_total] requests={combined_requests} input={combined_input_tokens} "
        f"(cached={combined_cached_input_tokens}) output={combined_output_tokens} "
        f"cost=${combined_cost_usd:.6f}"
    ),
    file=sys.stderr,
    flush=True,
)

usage_summary = {
    "model": args.model,
    "requests": combined_requests,
    "input_tokens": combined_input_tokens,
    "cached_input_tokens": combined_cached_input_tokens,
    "non_cached_input_tokens": combined_non_cached_input,
    "output_tokens": combined_output_tokens,
    "estimated_cost_usd": combined_cost_usd,
    "pricing_per_million": {
        "input": args.price_input_per_m or model_pricing["input"],
        "cached_input": args.price_cached_input_per_m or model_pricing["cached_input"],
        "output": args.price_output_per_m or model_pricing["output"],
    },
    "pricing_source": "auto_detected" if not any([args.price_input_per_m, args.price_cached_input_per_m, args.price_output_per_m]) else "mixed",
}

_atomic_json_dump(
    pathlib.Path(args.output_file).with_suffix(".usage.json"),
    usage_summary,
)
print(_color(f"[save] Wrote final {args.output_file}.usage.json. est_cost=${combined_cost_usd:.6f}", CLR_MAGENTA), file=sys.stderr, flush=True)

# Clean up HTTP resources to avoid leaking file descriptors
try:
    client.close()
except Exception:
    pass