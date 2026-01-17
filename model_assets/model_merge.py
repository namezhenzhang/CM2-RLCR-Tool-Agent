from typing import List, Tuple, Dict
import re
import os
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from torch.distributed._tensor import DTensor, Shard, Placement


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")

def merge_model(local_dir, output_path):
    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 32
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)  
            break  
    assert world_size, "No model file with the proper format"

    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'), map_location='cpu')
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor)
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    assert mesh_dim_names in (
        ('fsdp',),
    ), f'Unsupported mesh_dim_names {mesh_dim_names}'

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:
                print("-"*30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == 'dp':
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print('Writing to local disk')
    hf_path = os.path.join(local_dir, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {output_path}')
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path, state_dict=state_dict)


def main(TRAIN_OUTPUT_DIR):
    local_dir = f"{TRAIN_OUTPUT_DIR}/actor"  
    # hf_path = f"{TRAIN_OUTPUT_DIR}/actor"  
    output_path = f"{TRAIN_OUTPUT_DIR}/actor".replace("checkpoints", "checkpoints_merged") 
    os.makedirs(output_path, exist_ok=True)
    merge_model(local_dir, output_path)

if __name__ == '__main__':
        # step = 60
    local_dir = f"/workspace/verl/checkpoints/checklist-exp/checklist_step-tis-newdata-cold-start-qwen3-4b-base-notag-keepthink_bs-128-n-32-c-1_4500-10000_ppo-minibatch-128-ppoepochs-1-kl-0.0001-ent-0.0-lr-1e-6-node-1-3-30B-A3B-Instruct-2507/global_step_55/actor"  # 这里需要替换为绝对路径
    hf_path = f"/workspace/verl/checkpoints/checklist-exp/checklist_step-tis-newdata-cold-start-qwen3-4b-base-notag-keepthink_bs-128-n-32-c-1_4500-10000_ppo-minibatch-128-ppoepochs-1-kl-0.0001-ent-0.0-lr-1e-6-node-1-3-30B-A3B-Instruct-2507/global_step_55/actor"  # 这里需要替换为绝对路径
    output_path = f"/workspace/verl/checkpoints_merged/checklist-exp/checklist_step-tis-newdata-cold-start-qwen3-4b-base-notag-keepthink_bs-128-n-32-c-1_4500-10000_ppo-minibatch-128-ppoepochs-1-kl-0.0001-ent-0.0-lr-1e-6-node-1-3-30B-A3B-Instruct-2507/global_step_55/actor"  # 这里需要替换为绝对路径

    TRAIN_OUTPUT_DIR = "/workspace/verl/checkpoints/checklist-exp-song-1109/drgrpo-tis-fix2-newdata-cold-start-qwen3-8b-base-notag-keepthink_bs-128-n-48-c-1_4000-10000_ppo-minibatch-128-ppoepochs-1-kl-0.001-ent-0.0-lr-3e-6-node-4-4-30B-A3B-Instruct-2507-1111/global_step_500"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=TRAIN_OUTPUT_DIR, help="FSDP checkpoints from VerL training")
    args = parser.parse_args() 
    main(args.checkpoint_dir)

    # Example: python run/model_merge.py --train_output_dir "/data/users/songwang/repos/Checklist-Agent-RL/verl/checkpoints/checklist-exp-song-1109/drgrpo-tis-fix2-newdata-cold-start-qwen3-8b-base-notag-keepthink_bs-128-n-48-c-1_4000-10000_ppo-minibatch-128-ppoepochs-1-kl-0.001-ent-0.0-lr-3e-6-node-4-4-30B-A3B-Instruct-2507-1111/global_step_500"

