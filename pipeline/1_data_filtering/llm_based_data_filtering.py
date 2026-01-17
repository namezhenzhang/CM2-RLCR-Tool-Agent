import os
import json
import time
import signal
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datasets import load_from_disk

from openai import OpenAI

DEFAULT_SYSTEM_PROMPT = """This is a simulated set of messages among a user, an assistant, and tools. The tools listed are the candidate tools. The assistant will first think (inside <think> and </think>; this part will be removed in post-processing and not shown to the user, and it should not be considered when judging whether there is an error), then decide whether to call a tool or produce a final response.

Does the logic for tool calling by the assistant follow the user's query, have no ambiguities, and can realistically occur in real scenarios? Are there any mistakes or flaws? Is it something that could exist in reality?
If there is no problem, answer true; if there is a problem, answer false.
You should be very strict.

Response format:
{
"Reasoning": string,
"NoError": true or false
}"""

stop_flag = False

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel call to the model to judge logical consistency for each record in the dataset and save incrementally.")
    parser.add_argument("--input", required=True, help="Input JSON data file path")
    parser.add_argument("--output", required=True, help="Output (passed data) save file")
    parser.add_argument("--raw-output", default=None, help="(Optional) Save intermediate status file for all entries (including failures/parsing failures)")
    parser.add_argument("--model", default="gpt-5", help="Model name to use")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent threads")
    parser.add_argument("--effort", type=str, default="medium", help="effort of gpt-5")
    parser.add_argument("--save-every", type=int, default=50, help="Incrementally save every N entries processed (completed)")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries for a single call")
    parser.add_argument("--retry-base-wait", type=float, default=2.0, help="Wait seconds for the first retry (exponential backoff)")
    parser.add_argument("--timeout-log", type=float, default=60.0, help="If no new completions for a long time, print progress log")
    parser.add_argument("--system-prompt-file", default=None, help="Custom system prompt file (uses built-in if not provided)")
    parser.add_argument("--n-samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_system_prompt(path: Optional[str]) -> str:
    if path is None:
        return DEFAULT_SYSTEM_PROMPT
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_data(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        ds = load_from_disk(path)
        data = []
        for i in range(len(ds)):
            sample = ds[i]
            uuid = sample.get("uuid")
            messages = sample.get("messages", [])
            metadata = json.loads(sample.get("metadata", "{}"))
            tools = metadata.get("tools", [])
            data.append({"uuid":uuid, "messages":messages,"tools":tools})
        return data
        

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None

def build_inputs(system_prompt: str, d: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = json.dumps(d["messages"], indent=2, ensure_ascii=False)
    tools = json.dumps(d["tools"], indent=2, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"# Messages:\n{messages}\n# Candidate Tools:\n{tools}"}
    ]

def call_model(client: OpenAI,
               model: str,
               system_prompt: str,
               d: Dict[str, Any],
               max_retries: int,
               retry_base_wait: float,
               index: int,
               effort: str) -> Dict[str, Any]:
    """
    Return structure:
    {
      "index": original index,
      "success": bool,
      "parsed": bool,
      "no_error": bool or None,
      "raw_text": model output,
      "reasoning": Reasoning when parsable or None,
      "error": error description or None
    }
    """
    inputs = build_inputs(system_prompt, d)
    attempt = 0
    uuid = d["uuid"]
    while attempt <= max_retries:
        if stop_flag:
            return {
                "index": index, "uuid": uuid, "success": False, "parsed": False, "no_error": None,
                "raw_text": None, "reasoning": None, "error": "stopped"
            }
        try:
            resp = client.responses.create(
                model=model,
                input=inputs,
                reasoning={"effort": effort},
                text={"verbosity": "medium"},
            )
            output_text = resp.output_text
            parsed = safe_json_parse(output_text)
            if parsed is None:
                return {
                    "index": index, "uuid": uuid,  "success": True, "parsed": False, "no_error": None,
                    "raw_text": output_text, "reasoning": None,
                    "error": "json_parse_failed"
                }
            no_error = bool(parsed.get("NoError"))
            reasoning = parsed.get("Reasoning")
            return {
                "index": index, "uuid" : uuid, "success": True, "parsed": True, "no_error": no_error,
                "raw_text": output_text, "reasoning": reasoning, "error": None
            }
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                return {
                    "index": index, "uuid" : uuid,  "success": False, "parsed": False, "no_error": None,
                    "raw_text": None, "reasoning": None,
                    "error": f"exception:{type(e).__name__}:{e}"
                }
            backoff = retry_base_wait * (2 ** (attempt - 1))
            time.sleep(min(backoff, 60))

def incremental_save(output_pass_path: str,
                     raw_output_path: Optional[str],
                     results_slot: List[Optional[Dict[str, Any]]],
                     original_data: List[Dict[str, Any]],
                     lock: threading.Lock):
    """
    Save:
    1. Filtered entries (NoError=True) -> output_pass_path
    2. All processed structures (including failures) (optional) -> raw_output_path
    """
    with lock:
        passed = []
        raw_list = []
        for idx, r in enumerate(results_slot):
            if r is None:
                continue
            if r.get("no_error"):
                passed.append(original_data[idx])
            if raw_output_path:
                raw_list.append(r)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_pass_path)), exist_ok=True)
        if raw_output_path:
            os.makedirs(os.path.dirname(os.path.abspath(raw_output_path)), exist_ok=True)

        # Save passed entries
        tmp_pass = output_pass_path + ".tmp"
        with open(tmp_pass, "w", encoding="utf-8") as f:
            json.dump(passed, f, ensure_ascii=False, indent=2)
        os.replace(tmp_pass, output_pass_path)
        # Save original entries
        if raw_output_path:
            tmp_raw = raw_output_path + ".tmp"
            with open(tmp_raw, "w", encoding="utf-8") as f:
                json.dump(raw_list, f, ensure_ascii=False, indent=2)
            os.replace(tmp_raw, raw_output_path)

def main():
    args = parse_args()
    system_prompt = load_system_prompt(args.system_prompt_file)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable OPENAI_API_KEY not detected, please export OPENAI_API_KEY=sk-xxx first")

    client = OpenAI(api_key=api_key)
    data = load_data(args.input)

    # Add sampling logic
    if args.n_samples > 0 and args.n_samples < len(data):
        import numpy as np
        # Use fixed random seed to ensure reproducibility
        indices = np.random.RandomState(seed=args.seed).choice(len(data), args.n_samples, replace=False).tolist()
        data = [data[i] for i in indices]
        print(f"[Info] Randomly sampled {args.n_samples} entries for processing")
    
    total = len(data)

    print(f"[Info] Data size: {total}")
    print(f"[Info] Using model: {args.model}, workers: {args.workers}")

    results_slot: List[Optional[Dict[str, Any]]] = [None] * total

    lock = threading.Lock()
    completed = 0
    succeeded = 0
    passed = 0
    last_save_completed = 0
    last_progress_time = time.time()

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx, item in enumerate(data):
            futures.append(
                executor.submit(
                    call_model,
                    client,
                    args.model,
                    system_prompt,
                    item,
                    args.max_retries,
                    args.retry_base_wait,
                    idx,
                    args.effort
                )
            )

        for fut in as_completed(futures):
            if stop_flag:
                print("[Info] Stop signal received, stopping waiting for remaining tasks.")
                break
            result = fut.result()
            idx = result["index"]
            with lock:
                if results_slot[idx] is None:
                    results_slot[idx] = result
                    completed += 1
                    if result["success"]:
                        succeeded += 1
                        if result["no_error"]:
                            passed += 1
            now = time.time()

            # Print brief log for single entry
            if result["success"]:
                status = "OK" if result["parsed"] else "UNPARSED"
            else:
                status = "FAIL"
            print(f"[{completed}/{total}] idx={idx} status={status} no_error={result.get('no_error')} err={result.get('error')}")

            # Save periodically
            if completed - last_save_completed >= args.save_every:
                incremental_save(args.output, args.raw_output, results_slot, data, lock)
                last_save_completed = completed
                print(f"[Save] Incrementally saved (completed={completed}, passed={passed})")

            # Print progress if no output for a while
            if now - last_progress_time > args.timeout_log:
                print(f"[Progress] completed={completed}/{total} succeeded={succeeded} passed={passed}")
                last_progress_time = now

    # Final save before finishing
    incremental_save(args.output, args.raw_output, results_slot, data, lock)
    print(f"[Done] Total: completed={completed}, succeeded={succeeded}, passed={passed}. Results saved to {args.output}")
    if args.raw_output:
        print(f"[Done] Raw details saved to {args.raw_output}")

    if stop_flag:
        print("[Warn] Task stopped in the middle (some data might not be processed).")


if __name__ == "__main__":
    main()