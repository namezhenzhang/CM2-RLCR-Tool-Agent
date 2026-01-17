import json
from pathlib import Path
from typing import Dict, Any, List, Set
from datasets import load_from_disk, Dataset
import argparse
from tqdm import tqdm
import numpy as np

# Reuse statistics computation to avoid divergence
from data_distribution import compute_statistics


# Hard-coded drop policy
# Apply ratios from ALL metrics; when multiple apply, combine as: 1 - Π(1 - r_i)
GLOBAL_DEFAULT_DROP_RATIO: float = 0.0  # when no mapping applies

# Per-metric ratio mappings (keys are the metric values to drop and their ratios)

DROP_RATIO_BY_MAX_ASSISTANT_STEPS: Dict[int, float] = {
    1: 0.3,
    2: 0.0
}
DROP_RATIO_BY_MAX_PARALLEL_TOOL_CALLS: Dict[int, float] = {
    1: 0.0
}
DROP_RATIO_BY_NUM_ASSISTANT_MESSAGES: Dict[int, float] = {
    1: 0.2,
    2: 0.3
}
DROP_RATIO_BY_NUM_CANDIDATE_TOOLS: Dict[int, float] = {
    1: 0.7,
    2: 0.4,
    3: 0.0,
    6: 0.0,
    10: 0.0
}
DROP_RATIO_BY_NUM_SELECTED_TOOLS: Dict[int, float] = {
    1: 0.4
}

DROP_RATIO_BY_NUM_TURNS: Dict[int, float] = {
    1: 0.4,
    3: 0.0
}
DROP_RATIO_BY_TOOL_CALLS: Dict[int, float] = {
    1: 0.3,
    2: 0.3,
    3: 0.0
}

# Unified mapping for convenient lookup by metric name
DROP_RATIOS_BY_METRIC: Dict[str, Dict[int, float]] = {
    "tool_calls_total": DROP_RATIO_BY_TOOL_CALLS,
    "max_parallel_tool_calls": DROP_RATIO_BY_MAX_PARALLEL_TOOL_CALLS,
    "max_assistant_steps": DROP_RATIO_BY_MAX_ASSISTANT_STEPS,
    "num_turns": DROP_RATIO_BY_NUM_TURNS,
    "num_selected_tools": DROP_RATIO_BY_NUM_SELECTED_TOOLS,
    "num_candidate_tools": DROP_RATIO_BY_NUM_CANDIDATE_TOOLS,
    "num_assistant_messages": DROP_RATIO_BY_NUM_ASSISTANT_MESSAGES,
}


# No base whitelist: all samples are eligible; dropping is controlled purely by per-metric ratio mappings


def load_stats_jsonl(stats_path: Path) -> Dict[str, Dict[str, Any]]:
    stats_by_uuid: Dict[str, Dict[str, Any]] = {}
    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uuid = obj.get("uuid")
            if uuid:
                stats_by_uuid[uuid] = obj
    return stats_by_uuid


def matches_criteria(stats: Dict[str, Any]) -> bool:
    # Consider all samples; if a value matches any mapping key, it contributes a drop ratio.
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "data/nemotron_post_training_dataset_v1/tool_calling_filtered_selected"),
    )
    parser.add_argument(
        "--stats_jsonl",
        type=str,
        default=str(Path(__file__).parent / "data/statistics.jsonl"),
        help="Optional statistics JSONL (uuid keyed). If missing or uuid not found, stats will be computed on-the-fly.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling matched samples to drop")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir.endswith(".json"):
        with open(dataset_dir,"r") as f:
            ds = json.load(f)
    else:
        ds = load_from_disk(dataset_dir)

    # ds = load_from_disk(args.dataset_dir)

    stats_by_uuid: Dict[str, Dict[str, Any]] = {}
    stats_path = Path(args.stats_jsonl) if args.stats_jsonl else None
    if stats_path and stats_path.exists():
        stats_by_uuid = load_stats_jsonl(stats_path)
        print(f"Loaded {len(stats_by_uuid)} precomputed stats from {stats_path}")
    else:
        print("No precomputed stats found. Computing on-the-fly...")

    matched_indices: List[int] = []
    matched_stats_by_index: Dict[int, Dict[str, Any]] = {}
    missing_stats_count = 0

    for idx in tqdm(range(len(ds)), desc="Filtering dataset"):
        sample = ds[int(idx)]
        uuid = sample.get("uuid")

        stats = None
        if uuid and uuid in stats_by_uuid:
            stats = stats_by_uuid[uuid]
        else:
            # Compute on the fly
            messages = sample.get("messages", [])
            if dataset_dir.endswith(".json"):
                tools = sample.get("tools", [])
            else:
                metadata_raw = sample.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else (metadata_raw or {})
                except Exception:
                    metadata = {}
                tools = metadata.get("tools", [])
            stats = compute_statistics(messages, tools)
            if uuid:
                missing_stats_count += 1

        if matches_criteria(stats):
            matched_indices.append(idx)
            matched_stats_by_index[idx] = stats
    total = len(ds)
    matched = len(matched_indices)
    print(
        f"Matched {matched} / {total} samples. "
        + (f"Computed stats for {missing_stats_count} missing uuids." if missing_stats_count else "")
    )

    if matched == 0:
        print("No samples matched the criteria. Exiting without writing output.")
        return

    # Decide drop on a per-sample basis using hard-coded ratios from all metrics
    rng = np.random.RandomState(seed=args.seed)
    drop_indices: Set[int] = set()
    for idx in matched_indices:
        s = matched_stats_by_index.get(idx, {})
        ratios: List[float] = []
        for metric_name, mapping in DROP_RATIOS_BY_METRIC.items():
            value = s.get(metric_name)
            if isinstance(value, int) and value in mapping:
                ratios.append(float(mapping[value]))
        if ratios:
            # Combine independent drop probabilities: 1 - product(1-r)
            keep_probs = [1.0 - max(0.0, min(1.0, r)) for r in ratios]
            combined_keep = 1.0
            for kp in keep_probs:
                combined_keep *= kp
            r = 1.0 - combined_keep
        else:
            r = float(GLOBAL_DEFAULT_DROP_RATIO)
        r = max(0.0, min(1.0, float(r)))
        if rng.rand() < r:
            drop_indices.add(idx)

    kept_indices = [i for i in range(total) if i not in drop_indices]
    kept = len(kept_indices)
    print(
        "Dropping "
        + str(len(drop_indices))
        + " matched samples using hard-coded ratios. Keeping "
        + str(kept)
        + " / "
        + str(total)
        + " samples."
    )
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if dataset_dir.endswith(".json"):
        subset = []
        for indice in kept_indices:
            subset.append(ds[indice])
        with open(output_dir,"w") as f:
            json.dump(subset,f,indent=2)
    else:
        subset = ds.select(kept_indices)
        subset.save_to_disk(str(output_dir))
    print(f"Saved filtered dataset to {output_dir}")


if __name__ == "__main__":
    main()


