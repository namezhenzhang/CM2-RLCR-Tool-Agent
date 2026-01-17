# Download the dataset
# from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split="tool_calling")
# ds.save_to_disk("/data/users/zhenzhang/dir1/training/data/nvidia_nemotron_v1_tool_calling")


from datasets import load_from_disk
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--template", type=str, choices=["qwen3", "xlam"], default="qwen3")
parser.add_argument("--n_samples", type=int, default=-1)
# split multi-turn dialogs into multiple single-turn examples (default: on)
group = parser.add_mutually_exclusive_group()
group.add_argument("--split_turns", dest="split_turns", action="store_true")
group.add_argument("--no_split_turns", dest="split_turns", action="store_false")
parser.set_defaults(split_turns=False)
parser.add_argument(
    "--split_assistant_per_user",
    action="store_true",
    help="If set, create one example for each assistant message after a user turn.",
)
# parser.set_defaults(split_assistant_per_user=True)
args = parser.parse_args()


current_file_path = Path(__file__).parent

with open(args.input, "r") as f:
    ds = json.load(f)

data_list = []
total_num = len(ds)
# random sample n_samples
if args.n_samples != -1:
    indices = np.random.RandomState(seed=42).choice(total_num, args.n_samples, replace=False).tolist()
else:
    indices = list(range(total_num))




for i in tqdm(indices):

    tools = ds[i]["tools"]

    data = []
    # merge consecutive tool messages
    messages = ds[i]["messages"]
    grouped_messages = []
    j = 0
    while j < len(messages):
        msg = messages[j]
        if msg["role"] == "tool":
            merged_contents = []
            merged_tool_calls = []
            while j < len(messages) and messages[j]["role"] == "tool":
                m = messages[j]
                m["role"] = "observation"
                merged_contents.append(m["content"])
                merged_tool_calls.extend(m.get("tool_calls", []))
                j += 1
            merged_content = ("\n</tool_response>\n<tool_response>\n").join(merged_contents)
            grouped_messages.append({"role": "observation", "content": merged_content, "tool_calls": merged_tool_calls})
        else:
            grouped_messages.append(messages[j])
            j += 1
    # helper to render assistant content with or without tool_calls
    def render_message_content(msg: Dict[str, Any], inject_tool_calls: bool, remove_think: bool = False) -> str:
        content = msg["content"]
        # strip hidden reasoning wrapped in <think>...</think>
        if remove_think:
            if isinstance(content, str):
                if "</think>" in content:
                    content = content.split("</think>", 1)[-1].lstrip("\n")
                elif "<think>" in content:
                    content = content.replace("<think>", "")

        tool_calls = msg.get("tool_calls", [])
        if inject_tool_calls and len(tool_calls) > 0:
            if content != "":
                content += "\n"
            for k, tool_call in enumerate(tool_calls):
                if "function" in tool_call:
                    function = tool_call["function"]
                else:
                    function = tool_call

                if ((k == 0 and content != "") or k > 0) and args.template == "qwen3":
                    content += "\n"

                if args.template == "xlam" and k == 0:
                    content += "["

                if args.template == "qwen3-coder":
                    content += f"\n<tool_call>\n<function='{function['name']}'>\n"
                    if "arguments" in function:
                        for key, value in json.loads(function["arguments"]).items():
                            content += f"<parameter='{key}'>\n{str(value)}\n</parameter>\n"
                    content += "</function>\n</tool_call>"
                elif args.template == "qwen3":
                    content += f"<tool_call>\n{{\"name\":\"{function['name']}\", \"arguments\": "
                    if "arguments" in function:
                        content += function["arguments"]
                    content += "}\n</tool_call>"
                elif args.template == "xlam":
                    content += f"{{\"name\":\"{function['name']}\", \"arguments\": "
                    if "arguments" in function:
                        content += function["arguments"]
                    content += "}"
                    if k < len(tool_calls) - 1:
                        content += ", "
                else:
                    raise ValueError(f"Template {args.template} not supported")

                if args.template == "xlam" and k == len(tool_calls) - 1:
                    content += "]"
        return content

    # normalize roles (tool -> observation)
    for d in grouped_messages:
        if d["role"] == "tool":
            d["role"] = "observation"

    if args.split_assistant_per_user:
        # split each assistant message after a user into its own example
        user_indices = [idx for idx, m in enumerate(grouped_messages) if m["role"] == "user"]
        if not user_indices:
            # fallback: no user message, keep as one sample using original behavior (inject tools everywhere)
            data = []
            for d in grouped_messages:
                if d["role"] == "observation":
                    data.append({"role": d["role"], "content": d["content"]})
                else:
                    data.append({"role": d["role"], "content": render_message_content(d, True)})
            data_list.append({"messages": data, "tools": json.dumps(tools)})
        else:
            for turn_idx, u_start in enumerate(user_indices):
                u_end = (user_indices[turn_idx + 1] - 1) if (turn_idx + 1 < len(user_indices)) else (len(grouped_messages) - 1)
                # find all assistant messages within this user turn
                assistant_indices = [idx for idx in range(u_start, u_end + 1) if grouped_messages[idx]["role"] == "assistant"]
                for a_idx in assistant_indices:
                    example_messages: List[Dict[str, str]] = []
                    # include history up to and including this assistant message
                    for idx_msg, msg in enumerate(grouped_messages[: a_idx + 1]):
                        if idx_msg < u_start:
                            example_messages.append({"role": msg["role"], "content": render_message_content(msg, True, True)})
                        else:
                            example_messages.append({"role": msg["role"], "content": render_message_content(msg, True)})
                    data_list.append({"messages": example_messages, "tools": json.dumps(tools)})
    elif args.split_turns:
        # find user turn starts
        user_indices = [idx for idx, m in enumerate(grouped_messages) if m["role"] == "user"]
        if not user_indices:
            # fallback: no user message, keep as one sample using original behavior (inject tools everywhere)
            data = []
            for d in grouped_messages:
                if d["role"] == "observation":
                    # keep all observations since we are not splitting
                    data.append({"role": d["role"], "content": d["content"]})
                else:
                    inject = True
                    data.append({"role": d["role"], "content": render_message_content(d, inject)})
            data_list.append({"messages": data, "tools": json.dumps(tools)})
        else:
            # create one example per user turn up to the last assistant before the next user
            for turn_idx, u_start in enumerate(user_indices):
                u_end = (user_indices[turn_idx + 1] - 1) if (turn_idx + 1 < len(user_indices)) else (len(grouped_messages) - 1)
                # build messages up to u_end (inclusive)
                example_messages: List[Dict[str, str]] = []
                for idx_msg, msg in enumerate(grouped_messages[: u_end + 1]):
                    if idx_msg < u_start:
                        example_messages.append({"role": msg["role"], "content": render_message_content(msg, True, True)})
                    else:
                        # within current turn: keep all, inject tools for assistant
                        if msg["role"] == "observation":
                            example_messages.append({"role": msg["role"], "content": msg["content"]})
                        else:
                            example_messages.append({"role": msg["role"], "content": render_message_content(msg, True)})
                data_list.append({"messages": example_messages, "tools": json.dumps(tools)})
    else:
        # original single-sample behavior: include all, inject tool calls wherever present
        data = []
        for d in grouped_messages:
            if d["role"] == "observation":
                data.append({"role": d["role"], "content": d["content"]})
            else:
                data.append({"role": d["role"], "content": render_message_content(d, True)})
        data_list.append({"messages": data, "tools": json.dumps(tools)})

# save data_list
if args.split_assistant_per_user:
    suffix = "_split_assistant"
elif args.split_turns:
    suffix = "_split"
else:
    suffix = ""
output_path = args.output
# output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(data_list, f, indent=4)



 





