import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from datasets import load_from_disk
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def group_turns(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group messages into conversational turns anchored by user messages.

    A turn starts at a user message and includes subsequent assistant/tool messages
    until the next user or end.
    """
    turns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            if current:
                turns.append(current)
                current = []
            current.append(m)
        else:
            if not current:
                # Skip leading non-user content but still collect sequentially
                current = []
            current.append(m)
    if current:
        turns.append(current)
    return turns


def compute_statistics(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-sample statistics.

    - num_turns: number of conversational turns (user-anchored)
    - max_tool_calls_after_user: maximum total tool calls an assistant made after a user within a turn
    - max_parallel_tool_calls: maximum number of tool responses directly following an assistant with tool_calls
    - num_candidate_tools: number of candidate tools in metadata (tools list length)
    - num_selected_tools: number of unique tools actually used (by name) across tool_calls
    - max_assistant_steps: maximum number of assistant messages between a user and the next user
    - num_assistant_messages: total number of assistant messages in the conversation
    """
    # Turns
    turns = group_turns(messages)
    num_turns = len(turns)

    # Candidate tools
    num_candidate_tools = len(tools or [])

    # Count assistant messages
    num_assistant_messages = sum(1 for m in messages if m.get("role") == "assistant")

    max_parallel_tool_calls = 0
    tool_calls_total = 0
    used_tool_unique = set()

    # Max assistant steps between a user and the next user
    max_assistant_steps = 0
    in_segment = False
    assistant_count_in_segment = 0
    for m in messages:
        role = m.get("role")
        if role == "user":
            if in_segment:
                if assistant_count_in_segment > max_assistant_steps:
                    max_assistant_steps = assistant_count_in_segment
            in_segment = True
            assistant_count_in_segment = 0
        elif role == "assistant":
            if in_segment:
                assistant_count_in_segment += 1
    if in_segment:
        if assistant_count_in_segment > max_assistant_steps:
            max_assistant_steps = assistant_count_in_segment

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", []) or []
            if tool_calls:
                # Total tool calls issued by this assistant
                max_parallel_tool_calls = max(max_parallel_tool_calls, len(tool_calls))
                tool_calls_total += len(tool_calls)

                # Collect tool names used
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name")
                    if name:
                        used_tool_unique.add(name)
        i += 1

    num_selected_tools = len(used_tool_unique)

    return {
        "num_turns": num_turns,
        "tool_calls_total": tool_calls_total,
        "max_parallel_tool_calls": max_parallel_tool_calls,
        "num_candidate_tools": num_candidate_tools,
        "num_selected_tools": num_selected_tools,
        "max_assistant_steps": max_assistant_steps,
        "num_assistant_messages": num_assistant_messages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../data/nemotron_post_training_dataset_v1/tool_calling_rule_based_filtered")
    parser.add_argument("--output", type=str, default="../data/nemotron_post_training_dataset_v1/tool_calling_rule_based_filtered/data_distribution.jsonl")
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of random samples to process; <=0 means all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--plot_metric", type=str, default="", help="Metric(s) to plot: name, comma-separated list, or 'all'; empty disables")
    parser.add_argument("--plot_output", type=str, default="", help="Output path. For multiple metrics, use a directory or include '{metric}' in the filename. Defaults to data/{metric}_hist.png")
    parser.add_argument("--plot_bins", type=int, default=-1, help="Number of bins; <=0 means adaptive (aim one per integer where possible)")
    args = parser.parse_args()

    if Path(args.dataset_dir).is_dir():
        ds = load_from_disk(args.dataset_dir)
    elif args.dataset_dir.endswith(".json"):
        with open(args.dataset_dir, "r") as f:
            ds = json.load(f)


    # Select indices
    if args.n_samples and args.n_samples > 0 and args.n_samples < len(ds):
        rng = np.random.RandomState(seed=args.seed)
        indices = rng.choice(len(ds), args.n_samples, replace=False).tolist()
    else:
        indices = list(range(len(ds)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Accumulate values for optional plotting
    values_by_metric = {
        "num_turns": [],
        "tool_calls_total": [],
        "max_parallel_tool_calls": [],
        "num_candidate_tools": [],
        "num_selected_tools": [],
        "max_assistant_steps": [],
        "num_assistant_messages": [],
    }

    with output_path.open("w", encoding="utf-8") as f:
        for idx in tqdm(indices, desc="Computing statistics"):
            sample = ds[int(idx)]
            if Path(args.dataset_dir).is_dir():
                
                uuid = sample.get("uuid")
                messages = sample.get("messages", [])
                metadata = json.loads(sample.get("metadata", "{}"))
                tools = metadata.get("tools", [])
            else:
               uuid = sample["uuid"]
               messages = sample["messages"]
               tools = sample["tools"]


            stats = compute_statistics(messages, tools)
            record = {"uuid": uuid, **stats}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Collect for plotting
            for key in values_by_metric.keys():
                value = stats.get(key)
                if isinstance(value, (int, float)):
                    values_by_metric[key].append(value)

    print(f"Wrote statistics for {len(indices)} samples to {output_path}")

    # Optional plotting
    if args.plot_metric:
        available_metrics = list(values_by_metric.keys())
        requested = [m.strip() for m in args.plot_metric.split(",") if m.strip()] if args.plot_metric != "all" else available_metrics
        # Validate
        for m in requested:
            if m not in values_by_metric:
                available = ", ".join(sorted(available_metrics))
                raise ValueError(f"Unknown plot_metric '{m}'. Available: {available} or 'all'")

        # Helper to resolve output path per metric
        def resolve_output_path(metric_name: str, multiple: bool) -> Path:
            if args.plot_output:
                raw = args.plot_output
                if "{metric}" in raw:
                    return Path(raw.format(metric=metric_name))
                p = Path(raw)
                known_exts = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
                if p.suffix.lower() in known_exts and not multiple:
                    return p
                # Treat as directory or base path; put per-metric files inside
                return p / f"{metric_name}_hist.png"
            return args.plot_output / f"{metric_name}_hist.png"

        multiple = len(requested) > 1

        # Helper for adaptive bins
        def compute_adaptive_bins(values: List[float], max_unique: int = 200):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 10
            # If values are integers, try one bin per integer using half-offset edges
            if np.all(np.abs(arr - np.round(arr)) < 1e-9):
                vmin = int(np.min(arr))
                vmax = int(np.max(arr))
                unique_vals = np.unique(arr).size
                span = vmax - vmin + 1
                if unique_vals <= max_unique and span <= max_unique * 2:
                    return np.arange(vmin - 0.5, vmax + 1.5, 1.0)
            # Otherwise use Freedman–Diaconis rule, fallback to sqrt rule
            q25, q75 = np.percentile(arr, [25, 75])
            iqr = q75 - q25
            data_min, data_max = float(np.min(arr)), float(np.max(arr))
            if iqr > 0:
                bin_width = 2.0 * iqr * (arr.size ** (-1.0 / 3.0))
                if bin_width > 0:
                    num_bins = int(np.ceil((data_max - data_min) / bin_width))
                    return max(num_bins, 1)
            # Fallback
            num_bins = int(max(1, np.ceil(np.sqrt(arr.size))))
            return num_bins
        for metric in requested:
            values = values_by_metric[metric]
            if not values:
                print(f"No values collected for metric '{metric}', skipping plot")
                continue

            plot_output = resolve_output_path(metric, multiple)
            plot_output.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 5))
            bins_to_use = args.plot_bins if args.plot_bins and args.plot_bins > 0 else compute_adaptive_bins(values)
            plt.hist(values, bins=bins_to_use, color="#4C78A8", edgecolor="white")
            plt.title(f"Histogram of {metric}")
            plt.xlabel(metric)
            plt.ylabel("Count")
            plt.grid(True, axis="y", alpha=0.2)
            plt.tight_layout()
            plt.savefig(plot_output)
            plt.close()
            print(f"Saved plot to {plot_output}")


if __name__ == "__main__":
    main()
