from datasets import load_from_disk, load_dataset
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import re
import warnings
from filter.tool_schema import validate_data_sample
from filter.role_order import validate_role_order
from filter.tool_call_match import validate_tool_call_response_match
from filter.content import validate_no_tool_response_in_assistant_message, validate_non_empty_content, validate_think_tags
from filter.json_validation import validate_tool_response_json_valid
from filter.duplicate_tools import validate_no_duplicate_tools_by_schema, validate_no_duplicate_tools_by_name
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure logger to show line numbers with colors
handler = logging.StreamHandler()
formatter = logging.Formatter('\033[92m%(asctime)s\033[0m - \033[94m%(name)s\033[0m - \033[93m%(levelname)s\033[0m - \033[95m%(filename)s:%(lineno)d\033[0m - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def filter_data(ds, indices, args):
    valid_indices = []
    error_type = {
        "tool_schema": 0,
        "role_order": 0,
        "tool_call_response_match": 0,
        "no_tool_response_in_assistant_message": 0,
        "tool_response_json_valid": 0,
        "duplicate_tools_by_schema": 0,
        "duplicate_tools_by_name": 0,
        "non_empty_content": 0,
        "think_tags": 0,
    }
    for i in tqdm(indices):
        data = ds[i]
        uuid = data["uuid"]
        messages = data["messages"]
        metadata = json.loads(data["metadata"])
        tools = metadata["tools"]

        data = {
            "messages": messages,
            "tools": tools,
        }
        if_error = False
        if args.tool_schema:
            valid, error = validate_data_sample(data)
            if not valid:
                # logger.warning(f"{error}")
                error_type["tool_schema"] += 1
                if_error = True
        if args.role_order:
            valid, error = validate_role_order(messages)
            if not valid:
                # logger.warning(f"{error}")
                error_type["role_order"] += 1
                if_error = True
        if args.tool_call_response_match:
            valid, error = validate_tool_call_response_match(messages)
            if not valid:
                # logger.warning(f"{error}")
                error_type["tool_call_response_match"] += 1
                if_error = True
        if args.no_tool_response_in_assistant_message:
            valid, error = validate_no_tool_response_in_assistant_message(messages)
            if not valid:
                # logger.warning(f"{error}")
                error_type["no_tool_response_in_assistant_message"] += 1
                if_error = True
        if args.tool_response_json_valid:
            valid, error = validate_tool_response_json_valid(messages)
            if not valid:
                # logger.warning(f"{error}")
                error_type["tool_response_json_valid"] += 1
                if_error = True
        if hasattr(args, "duplicate_tools_by_schema") and args.duplicate_tools_by_schema:
            valid, error = validate_no_duplicate_tools_by_schema(tools)
            if not valid:
                # logger.warning(f"{error}")
                error_type["duplicate_tools_by_schema"] += 1
                if_error = True
        if hasattr(args, "duplicate_tools_by_name") and args.duplicate_tools_by_name:
            valid, error = validate_no_duplicate_tools_by_name(tools)
            if not valid:
                # logger.warning(f"{error}")
                error_type["duplicate_tools_by_name"] += 1
                if_error = True
        if args.non_empty_content:
            valid, error = validate_non_empty_content(messages)
            if not valid:
                # logger.warning(f"{error}")
                error_type["non_empty_content"] += 1
                if_error = True
        if args.think_tags:
            valid, error = validate_think_tags(messages)
            if not valid:
                # logger.warning(f"{error}")
                error_type["think_tags"] += 1
                if_error = True
        if not if_error:
            valid_indices.append(i)
    print(error_type)
    print(f"Filtered {len(valid_indices)} samples")
    return valid_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--tool_schema", action="store_true")
    parser.add_argument("--role_order", action="store_true")
    parser.add_argument("--tool_call_response_match", action="store_true")
    parser.add_argument("--no_tool_response_in_assistant_message", action="store_true")
    parser.add_argument("--tool_response_json_valid", action="store_true")
    parser.add_argument("--duplicate_tools_by_schema", action="store_true")
    parser.add_argument("--duplicate_tools_by_name", action="store_true")
    parser.add_argument("--non_empty_content", action="store_true")
    parser.add_argument("--think_tags", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="../data/nemotron_post_training_dataset_v1/tool_calling")
    parser.add_argument("--output_dir", type=str, default="../data/nemotron_post_training_dataset_v1/tool_calling_filtered")
    args = parser.parse_args()

    current_file_path = Path(__file__).parent

    ds = load_dataset("parquet", data_files=f"{args.dataset_dir}/**/*.parquet", split="train")

    if args.n_samples > len(ds):
        args.n_samples = len(ds)
        indices = list(range(len(ds)))
        logger.warning(f"--n_samples ({args.n_samples}) cannot exceed dataset size ({len(ds)})")
    elif args.n_samples <= 0:
        indices = list(range(len(ds)))
        logger.warning(f"--n_samples <= 0, will process all samples")
    else:
        indices = np.random.RandomState(seed=42).choice(len(ds), args.n_samples, replace=False).tolist()

    filtered_indices = filter_data(ds, indices, args)

    filtered_ds = ds.select(filtered_indices)
    filtered_ds.save_to_disk(args.output_dir)


