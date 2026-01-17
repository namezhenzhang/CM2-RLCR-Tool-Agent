import argparse
import json
from pathlib import Path
from typing import Any, List

import datasets


def load_json_records(input_path: Path) -> List[dict]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError("Expected top-level JSON array or an object with 'data' list")


def maybe_parse_tools(records: List[dict], parse_tools: bool) -> List[dict]:
    if not parse_tools:
        return records
    fixed: List[dict] = []
    for item in records:
        new_item: dict[str, Any] = dict(item)
        tools_val = new_item.get("tools")
        if isinstance(tools_val, str):
            try:
                new_item["tools"] = json.loads(tools_val)
            except Exception:
                # Keep original string if it fails to parse
                pass
        fixed.append(new_item)
    return fixed


def _sanitize_empty_dicts(value: Any) -> Any:
    """Recursively convert empty dicts to None to avoid Arrow empty-struct errors.

    This prevents schemas like struct<analyze_stock: struct<>> which Parquet cannot write.
    """
    if isinstance(value, dict):
        if len(value) == 0:
            return None
        # Recurse into nested structures
        return {k: _sanitize_empty_dicts(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_empty_dicts(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert annotated checklist JSON to Parquet")
    parser.add_argument(
        "--input",
        type=str,
        default="dir4/workspace/preprocess_data/annotated_checklist/nvidia_nemotron_v1_tool_calling_qwen3_checklist.json",
        help="Path to the input JSON file (top-level array of records)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dir4/workspace/preprocess_data/annotated_checklist/nvidia_nemotron_v1_tool_calling_qwen3_checklist.parquet",
        help="Path to the output Parquet file",
    )
    parser.add_argument(
        "--parse-tools-json",
        action="store_true",
        help="If set, parse the 'tools' field from JSON string to structured JSON",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="nvidia_nemotron_checklist",
        help="Value to fill into 'data_source' for each record if missing (required by training pipeline)",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=100,
        help="Number of samples to save as val split (taken from the start)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_json_records(input_path)
    records = maybe_parse_tools(records, args.parse_tools_json)

    # Ensure required fields exist for downstream trainers
    for r in records:
        r.setdefault("data_source", args.data_source)

    # Also write the full dataset to the original output path (no split suffix)
    full_ds = datasets.Dataset.from_list(records)
    print(full_ds)
    full_ds.to_parquet(str(output_path))
    print(f"Wrote Full Parquet ({len(full_ds)}): {output_path}")

    # Split into val/train
    n_test = max(0, int(args.n_val))
    val_records = records[:n_test]
    train_records = records[n_test:]

    # Derive output file paths with _train/_val suffixes
    base_path = output_path
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    train_path = base_path.parent / f"{base_path.name}_train.parquet"
    val_path = base_path.parent / f"{base_path.name}_val.parquet"
    full_path = base_path.parent / f"{base_path.name}.parquet"

    # Write train split if non-empty
    if len(train_records) > 0:
        train_ds = datasets.Dataset.from_list(train_records)
        print(train_ds)
        train_ds.to_parquet(str(train_path))
        print(f"Wrote Train Parquet ({len(train_ds)}): {train_path}")
    else:
        print("No train records to write.")

    # Write val split if non-empty
    if len(val_records) > 0:
        val_ds = datasets.Dataset.from_list(val_records)
        print(val_ds)
        val_ds.to_parquet(str(val_path))
        print(f"Wrote Val Parquet ({len(val_ds)}): {val_path}")
    else:
        print("No val records to write.")
    
    if len(records) > 0:
        full_ds = datasets.Dataset.from_list(records)
        print(full_ds)
        full_ds.to_parquet(str(full_path))
        print(f"Wrote Full Parquet ({len(full_ds)}): {full_path}")
    else:
        print("No full records to write.")

    # dataset = datasets.load_dataset("parquet", data_files=output_path)["train"]
    # print(dataset)


if __name__ == "__main__":
    main()


