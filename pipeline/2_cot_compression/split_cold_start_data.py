import json
from datasets import load_from_disk
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import uuid as generate_uuid
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--rl_output", type=str, default=None)
    parser.add_argument("--cold_start_output", type=str, default=None)


    args = parser.parse_args()

    if args.rl_output:
        Path(args.rl_output).parent.mkdir(parents=True, exist_ok=True)
    if args.cold_start_output:
        Path(args.cold_start_output).parent.mkdir(parents=True, exist_ok=True)

    current_file_path = Path(__file__).parent

    with open(args.input, 'r') as f:
        data = json.load(f)

    print(f"Original dataset size: {len(data)}")
    print(f"Cold start dataset size: {args.n_samples}")
    print(f"RL Training dataset size: {len(data)-args.n_samples}")
    
    # Randomly select n_samples from the filtered dataset
    if len(data) < args.n_samples:
        print(f"Warning: Dataset has only {len(data)} samples, but {args.n_samples} were requested")
        raise ValueError
    else:
        random.seed(42)  # 设置随机种子以确保结果可复现
        random.shuffle(data)  # 随机打乱数据
        selected_ds = data[:args.n_samples]  # 选择前 n_samples 个样本
        not_selected_ds = data[args.n_samples:]
    
    for d in not_selected_ds:
        if 'uuid' not in d:
            d['uuid'] = str(generate_uuid.uuid4())
    with open(args.rl_output,"w") as f:
        json.dump(not_selected_ds,f,indent=2)

    # Save the selected dataset
    formated_data = []
    for d in tqdm(selected_ds):
        uuid = d.get('uuid', str(generate_uuid.uuid4()))
        message = d['messages']
        tools = d['tools']
        formated_message = []
        for m in message:
            if m['role'] == 'user':
                formated_message.append({
                    'role': 'user',
                    'content': m['content'],
                    'tool_calls': []
                })
            elif m['role'] == 'assistant':
                thinking = m['content'].split('<think>')[1].split('</think>')[0].strip()
                if len(m['tool_calls'])== 0:
                    content_after_think = m['content'].split('</think>')[1].strip() if '</think>' in m['content'] else m['content']
                else:
                    content_after_think = ""
                
                formated_message.append({
                    'role': 'assistant',
                    'content': {
                        'thinking': thinking,
                        'reply': content_after_think
                    },
                    'tool_calls': m['tool_calls']
                })
            elif m['role'] == 'tool':
                formated_message.append({
                    'role': 'tool',
                    'content': m['content'],
                    'tool_calls': []
                })
            else:
                raise ValueError(f"Unknown role: {m['role']}")
        formated_data.append({
            'uuid': uuid,
            'messages': formated_message,
            'tools': tools
        })

    with open(args.cold_start_output, 'w') as f:
        json.dump(formated_data, f, indent=2)
