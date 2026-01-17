from openai import OpenAI
import os
import pathlib
import json
import re
import sys
from tqdm import tqdm
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import signal

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


def get_model_pricing(model_name: str) -> dict:
    """Get pricing configuration for a model, with fallbacks for similar names"""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    model_lower = model_name.lower().replace("_", "-").replace(" ", "-")
    for key in MODEL_PRICING:
        if key.lower().replace("_", "-").replace(" ", "-") == model_lower:
            return MODEL_PRICING[key]
    for key in MODEL_PRICING:
        key_clean = key.lower().replace("-", "").replace("_", "").replace(" ", "")
        model_clean = model_lower.replace("-", "").replace("_", "").replace(" ", "")
        if key_clean == model_clean:
            return MODEL_PRICING[key]
    print(_color(f"Warning: No pricing found for model '{model_name}', using GPT-5 pricing as default", CLR_YELLOW), file=sys.stderr)
    return MODEL_PRICING["gpt-5"]


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
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    stripped = _strip_code_fences(text)
    if stripped != text:
        try:
            json.loads(stripped)
            return stripped
        except Exception:
            pass

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


def _parse_result_json(text: str, expected_items: int) -> list[dict]:
    """Parse model output and validate minimal expectations.

    Expected format: array of objects, each having key 'thinking' (string).
    The list length must equal expected_items.
    """
    json_text = _extract_json_string(text)
    data = json.loads(json_text)
    if not isinstance(data, list):
        raise ValueError("Top-level output must be an array")
    if len(data) != expected_items:
        raise ValueError(f"assistant message count mismatch: parsed={len(data)} vs expected={expected_items}")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be an object")
        if "thinking" not in item:
            raise ValueError(f"Item {i} missing 'thinking'")
        if not isinstance(item["thinking"], str):
            raise ValueError(f"Item {i}.thinking must be string")
        if item["thinking"].strip() == "":
            raise ValueError(f"Item {i}.thinking must be non-empty after stripping")
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--input-json", type=str, default="../data/cold_start_data/cold_start_data.json")
parser.add_argument(
    "--output-file",
    type=str,
    default="../data/cold_start_data/cold_start_data_cot_compressed.json",
    help="File path for annotated JSON; defaults to ./cold_start_data/shorten_thinking_annotated.json next to this script",
)
parser.add_argument("--prompt-file", type=str, default="cot_compression_prompt_v1.txt", help="Prompt file")
parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
parser.add_argument("--model", type=str, default="gpt-5", help="Model name (pricing will be auto-detected)")
parser.add_argument("--price-input-per-m", type=float, default=None, help="Override input price (USD per 1M tokens)")
parser.add_argument("--price-cached-input-per-m", type=float, default=None, help="Override cached input price (USD per 1M tokens)")
parser.add_argument("--price-output-per-m", type=float, default=None, help="Override output price (USD per 1M tokens)")
parser.add_argument("--effort", type=str, default="medium", choices=["minimal", "low", "medium", "high"], help="Effort level for reasoning")
parser.add_argument("--verbosity", type=str, default="medium", choices=["low", "medium", "high"], help="Verbosity level for output")
parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("--retry", type=int, default=3)
parser.add_argument("--save_every", type=int, default=1000, help="Save partial results every N items")
parser.add_argument("--save_every_seconds", type=int, default=120, help="Save partial results every N seconds")
parser.add_argument("--enable-cache-control", action="store_true", help="Use structured system content; caching is automatic when prefix is identical")
parser.add_argument("--cache-warmup", action="store_true", help="Send a light warmup call to seed prompt cache")
parser.add_argument("--warmup-max-output-tokens", type=int, default=16, help="Max output tokens for warmup call (min 16)")

args = parser.parse_args()


# Set up client
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key == "":
    raise ValueError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)


# Resolve paths
current_folder = pathlib.Path(__file__).parent
output_path = pathlib.Path(args.output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)

if output_path.suffix:
    usage_path = output_path.with_suffix(".usage.json")
else:
    usage_path = output_path.with_name(output_path.name + ".usage.json")


# Load prompt
with open(args.prompt_file, "r") as f:
    system_prompt = f.read()


# Load data
with open(args.input_json, "r") as f:
    raw_dataset = json.load(f)

if not isinstance(raw_dataset, list):
    raise ValueError("Input JSON must be a list of samples")

indices = list(range(len(raw_dataset)))
if args.n_samples and args.n_samples > 0 and args.n_samples < len(indices):
    import numpy as np
    rng = np.random.RandomState(seed=args.seed)
    indices = rng.choice(len(raw_dataset), args.n_samples, replace=False).tolist()


"""
Resume support: if an existing annotated file is present, load it and skip
any items with matching UUIDs. Pre-seed result_list with existing results
so partial and final saves include them.
"""
existing_results: list[dict] = []
existing_uuids: set[str] = set()
try:
    existing_path = output_path
    if existing_path.exists():
        with open(existing_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                for it in loaded:
                    try:
                        u = it.get("uuid")
                        if isinstance(u, str):
                            existing_results.append(it)
                            existing_uuids.add(u)
                    except Exception:
                        continue
except Exception as e:
    print(_color(f"[resume] Failed to load existing results: {e}", CLR_YELLOW), file=sys.stderr, flush=True)

if existing_uuids:
    print(_color(f"Resuming: {len(existing_uuids)} already present", CLR_BLUE), flush=True)
else:
    print(_color(f"Annotating {len(indices)} samples", CLR_BLUE), flush=True)


# ---- Usage & Cost tracking helpers ----
model_pricing = get_model_pricing(args.model)
_PRICE_PER_INPUT_TOKEN = (args.price_input_per_m or model_pricing["input"]) / 1_000_000
_PRICE_PER_CACHED_INPUT_TOKEN = (args.price_cached_input_per_m or model_pricing["cached_input"]) / 1_000_000
_PRICE_PER_OUTPUT_TOKEN = (args.price_output_per_m or model_pricing["output"]) / 1_000_000

print(_color(f"Using model: {args.model}", CLR_BLUE), file=sys.stderr)
print(_color(f"Pricing (per 1M tokens): input=${model_pricing['input']:.3f}, cached=${model_pricing['cached_input']:.3f}, output=${model_pricing['output']:.3f}", CLR_CYAN), file=sys.stderr)
if args.price_input_per_m or args.price_cached_input_per_m or args.price_output_per_m:
    print(_color("Note: Some prices overridden via command line arguments", CLR_YELLOW), file=sys.stderr)
print("", file=sys.stderr)


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
    _atomic_json_dump(
        output_path,
        result_list,
    )
    print(_color("[save] Wrote partial shorten_thinking_annotated.json" if checkpoint else "[save] Wrote final shorten_thinking_annotated.json", CLR_MAGENTA), file=sys.stderr, flush=True)

    combined_requests = total_requests
    combined_input_tokens = total_input_tokens
    combined_cached_input_tokens = total_cached_input_tokens
    combined_output_tokens = total_output_tokens
    combined_non_cached_input = max(0, combined_input_tokens - combined_cached_input_tokens)
    combined_cost_usd = round(_estimate_cost_usd(total_input_tokens, total_cached_input_tokens, total_output_tokens), 6)
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
    if checkpoint:
        usage_summary_payload.update({
            "progress_completed": progress,
            "progress_total": PROGRESS_TOTAL,
            "checkpoint": True,
        })
    _atomic_json_dump(
        usage_path,
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
    """Build system content, optionally marking it as cacheable."""
    if args.enable_cache_control:
        return [
            {
                "type": "input_text",
                "text": system_prompt,
            }
        ]
    return system_prompt


def _count_assistant_messages(messages: list[dict]) -> int:
    try:
        return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
    except Exception:
        return 0


def _transform_messages_with_thinking(messages: list[dict], shortened_list: list[dict]) -> list[dict]:
    """Replace assistant messages' content with <think>wrapped</think> + original reply.

    Keep only keys: role, content, tool_calls for all messages.
    """
    out: list[dict] = []
    si = 0  # index into shortened_list aligned to assistant messages
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", []) if isinstance(msg.get("tool_calls", []), list) else []

        if role == "assistant":
            shortened_thinking = shortened_list[si].get("thinking", "").strip()
            si += 1
            new_content = f"<think>\n{shortened_thinking}\n</think>"
            if isinstance(content["reply"], str) and content["reply"].strip() != "":
                new_content = new_content + "\n\n" + content["reply"]
            out.append({
                "role": role,
                "content": new_content,
                "tool_calls": tool_calls,
            })
        else:
            out.append({
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
            })
    return out


def _process_one(i: int, item: dict) -> dict:
    local_requests = 0
    local_input_tokens = 0
    local_cached_input_tokens = 0
    local_output_tokens = 0
    attempt_cost_usd = 0.0
    last_error: Exception | None = None

    uuid = item.get("uuid")
    messages = item.get("messages", [])
    tools = item.get("tools", []) or []

    input_payload = {
        "tools": tools,
        "messages": messages,
    }

    inputs = [
        {"role": "system", "content": _build_system_content()},
        {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False, indent=4)},
    ]

    expected_assistant = _count_assistant_messages(messages)

    for attempt in range(args.retry):
        result = client.responses.create(
            model=args.model,
            input=inputs,
            reasoning={"effort": args.effort},
            text={"verbosity": args.verbosity},
        )
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
        try:
            parsed = _parse_result_json(output_text, expected_assistant)
            new_messages = _transform_messages_with_thinking(messages, parsed)
            print(_color(f"Turn {i} done, cost=${attempt_cost_usd:.6f}", CLR_GREEN), flush=True)
            return {
                "ok": True,
                "i": i,
                "attempt_cost_usd": attempt_cost_usd,
                "requests": local_requests,
                "input_tokens": local_input_tokens,
                "cached_input_tokens": local_cached_input_tokens,
                "output_tokens": local_output_tokens,
                "data": {
                    "uuid": uuid,
                    "tools": tools,
                    "messages": new_messages,
                },
            }
        except Exception as e:
            last_error = e
            print(
                _color(f"[parser] Parse failed (i={i}, attempt={attempt + 1}): {e}", CLR_RED),
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


# Optional cache warmup
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
    except Exception as e:
        print(_color(f"[cache_warmup] Failed: {e}", CLR_RED), file=sys.stderr, flush=True)


signal.signal(signal.SIGTERM, _handle_sigterm)

try:
    tasks: list[tuple[int, dict]] = []
    for i in indices:
        item = raw_dataset[int(i)]
        uuid = item.get("uuid")
        if uuid in existing_uuids:
            continue
        # Normalize shape to include tools if present elsewhere; default empty
        normalized = {
            "uuid": uuid,
            "messages": item.get("messages", []),
            "tools": item.get("tools", []),
        }
        tasks.append((i, normalized))

    PROGRESS_TOTAL = len(tasks)
    completed = 0
    last_save_time = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [executor.submit(_process_one, i, item) for (i, item) in tasks]
        with tqdm(total=PROGRESS_TOTAL) as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                with _agg_lock:
                    total_requests += res.get("requests", 0)
                    total_input_tokens += res.get("input_tokens", 0)
                    total_cached_input_tokens += res.get("cached_input_tokens", 0)
                    total_output_tokens += res.get("output_tokens", 0)
                    total_cost_usd = _estimate_cost_usd(total_input_tokens, total_cached_input_tokens, total_output_tokens)

                    if res.get("ok"):
                        result_list.append(res["data"])
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

_atomic_json_dump(output_path, result_list)
print(_color("[save] Wrote final shorten_thinking_annotated.json", CLR_MAGENTA), file=sys.stderr, flush=True)

combined_requests = total_requests
combined_input_tokens = total_input_tokens
combined_cached_input_tokens = total_cached_input_tokens
combined_output_tokens = total_output_tokens
combined_non_cached_input = max(0, combined_input_tokens - combined_cached_input_tokens)
combined_cost_usd = round(total_cost_usd, 6)
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
    usage_path,
    usage_summary,
)
print(_color(f"[save] Wrote final usage summary. est_cost=${combined_cost_usd:.6f}", CLR_MAGENTA), file=sys.stderr, flush=True)

try:
    client.close()
except Exception:
    pass


