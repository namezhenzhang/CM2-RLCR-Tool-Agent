import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

target_tokens = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>"]
semantic_tokens = [
    ["think", "begin"],
    ["think", "finish"],
    ["tool", "call", "start"],
    ["tool", "call", "end"],
    ["role", "enter"],
    ["role", "exit"]
]

add_noise = True
noise_scale = 0.005

# Whether to align each target token's final vector norm to
# the average norm of its source subtokens (separately for embedding / lm_head)
align_norm_with_source_subtokens = True

# combine_mode:
#  - "input_only": use only the mean of input-embedding subtokens
#  - "lmhead_only": use only the mean of lm_head subtokens (if untied)
#  - "avg": if untied, average the input and lm_head means; if tied, same as input_only
combine_mode = "avg"

device = "cpu"
dtype = torch.float32

def collect_subtoken_ids(tokenizer, semantic_unit_list):
    """
    semantic_unit_list, e.g. ["think", "start"].
    Tokenize each piece separately and then concatenate all subtoken ids.
    If you want to treat them as a single string, you can instead do:
        text = " ".join(semantic_unit_list)
        return tokenizer.encode(text, add_special_tokens=False)
    """
    all_ids = []
    for piece in semantic_unit_list:
        ids = tokenizer.encode(piece, add_special_tokens=False)
        all_ids.extend(ids)
    print("[DEBUG] semantic pieces:", semantic_unit_list, "-> sub_ids:", all_ids)
    return all_ids

def maybe_resize_for_new_tokens(model, tokenizer, old_vocab_size):
    """
    Safely resize embeddings (and possibly lm_head) after adding new tokens.
    """
    new_vocab_size = len(tokenizer)
    if new_vocab_size == old_vocab_size:
        return

    # Use HF's resize method to first extend input embeddings
    model.resize_token_embeddings(new_vocab_size)

    # Then check output embeddings (lm_head)
    try:
        out_layer = model.get_output_embeddings()
    except AttributeError:
        out_layer = None

    if out_layer is None:
        return

    # If embeddings are tied, the above resize already handled them; if not, we may need to extend manually
    in_weight = model.get_input_embeddings().weight
    out_weight = out_layer.weight

    if out_weight.shape[0] != new_vocab_size:
        # Manually extend
        old_out = out_weight.data
        hidden_dim = old_out.shape[1]
        new_param = torch.nn.Parameter(
            torch.empty(new_vocab_size, hidden_dim, device=old_out.device, dtype=old_out.dtype)
        )
        # Simple initialization; can use model.config.initializer_range
        init_range = getattr(model.config, "initializer_range", 0.02)
        torch.nn.init.normal_(new_param, mean=0.0, std=init_range)
        new_param[:old_out.shape[0]] = old_out
        out_layer.weight = new_param

def compute_mean_and_norms(weight_matrix, sub_ids):
    """
    Given a weight matrix (vocab, dim) and sub_ids, return:
      mean_vec: mean of subtoken vectors
      mean_norm: mean L2 norm of each subtoken (used for later alignment)
    """
    sub_embs = weight_matrix[sub_ids]  # shape [m, dim]
    mean_vec = sub_embs.mean(dim=0)
    sub_norms = sub_embs.norm(p=2, dim=-1)
    mean_norm = sub_norms.mean()
    return mean_vec, mean_norm

def align_vector_norm(vec, target_norm, eps=1e-8):
    cur_norm = vec.norm(p=2)
    if cur_norm > 0:
        scale = target_norm / (cur_norm + eps)
        return vec * scale
    return vec

def main(model_dir: str, save_dir: str):
    assert len(target_tokens) == len(semantic_tokens), "target_tokens and semantic_tokens must have the same length."

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)

    input_emb_module = model.get_input_embeddings()
    emb_weight = input_emb_module.weight  # (vocab, dim)
    old_vocab_size = emb_weight.shape[0]
    hidden_dim = emb_weight.shape[1]

    # Record whether embeddings are originally tied
    try:
        out_layer = model.get_output_embeddings()
    except AttributeError:
        out_layer = None

    if out_layer is None:
        raise RuntimeError("Model has no output layer (lm_head); cannot continue.")
    out_weight = out_layer.weight
    tied = (out_weight.data_ptr() == emb_weight.data_ptr())
    print(f"[INFO] Tied embeddings: {tied}")

    # 1. Check / add target tokens
    target_token_ids = []
    added_any = False
    for tok in target_tokens:
        if tok in tokenizer.get_vocab():
            tid = tokenizer.convert_tokens_to_ids(tok)
            target_token_ids.append(tid)
        else:
            tokenizer.add_tokens([tok], special_tokens=False)
            added_any = True
            tid = tokenizer.convert_tokens_to_ids(tok)
            target_token_ids.append(tid)

    # 2. If any new tokens were added, perform resize
    if added_any:
        maybe_resize_for_new_tokens(model, tokenizer, old_vocab_size)
        # Re-fetch latest weights
        input_emb_module = model.get_input_embeddings()
        emb_weight = input_emb_module.weight
        out_layer = model.get_output_embeddings()
        out_weight = out_layer.weight
        tied = (out_weight.data_ptr() == emb_weight.data_ptr())
        print(f"[INFO] After resize, tied={tied}, new vocab={emb_weight.shape[0]}")

    # 3. For each target token, build new vectors
    # Store results separately for embedding space and lm_head space
    new_vectors_emb = []
    new_vectors_out = []
    mean_source_norms_emb = []
    mean_source_norms_out = []
    source_subtoken_id_lists = []

    use_avg_mode = (combine_mode == "avg")
    use_input_only = (combine_mode == "input_only")
    use_lmhead_only = (combine_mode == "lmhead_only")

    if use_lmhead_only and tied:
        print("[WARN] combine_mode=lmhead_only but embeddings are tied; falling back to input_only.")
        use_lmhead_only = False
        use_input_only = True

    if use_avg_mode and tied:
        print("[INFO] combine_mode=avg with tied embeddings; equivalent to input_only.")

    for sem_list in semantic_tokens:
        sub_ids = collect_subtoken_ids(tokenizer, sem_list)
        if len(sub_ids) == 0:
            raise ValueError(f"Semantic unit {sem_list} is empty after tokenization.")
        source_subtoken_id_lists.append(sub_ids)

        with torch.no_grad():
            # 1) Input embedding space
            mean_vec_emb, mean_norm_emb = compute_mean_and_norms(emb_weight, sub_ids)

            # 2) lm_head space (if untied)
            if not tied:
                mean_vec_out, mean_norm_out = compute_mean_and_norms(out_weight, sub_ids)
            else:
                mean_vec_out, mean_norm_out = mean_vec_emb, mean_norm_emb  # placeholder for unified logic

            # 3) Build a "base vector" according to combine_mode (for later noise + alignment)
            if use_input_only:
                base_emb_vec = mean_vec_emb.clone()
                base_out_vec = mean_vec_out.clone() if not tied else base_emb_vec

            elif use_lmhead_only:
                # Use only the mean of lm_head subtokens
                base_emb_vec = mean_vec_emb.clone()  # still need a value for embeddings; could also use lm_head result
                base_out_vec = mean_vec_out.clone()
                # More aggressive option: also use lm_head mean for embeddings => base_emb_vec = mean_vec_out.clone()

            elif use_avg_mode:
                # Simply use the mean of the corresponding space:
                #   - emb: mean_vec_emb
                #   - lm_head: mean_vec_out (if untied) or mean_vec_emb (if tied)
                if not tied:
                    base_emb_vec = mean_vec_emb.clone()
                    base_out_vec = mean_vec_out.clone()
                else:
                    base_emb_vec = mean_vec_emb.clone()
                    base_out_vec = mean_vec_emb.clone()
            else:
                raise ValueError(f"Unknown combine_mode={combine_mode}")

            mean_source_norms_emb.append(mean_norm_emb)
            mean_source_norms_out.append(mean_norm_out if not tied else mean_norm_emb)

        new_vectors_emb.append(base_emb_vec)
        new_vectors_out.append(base_out_vec)

    new_vectors_emb = torch.stack(new_vectors_emb, dim=0)  # [N, dim]
    new_vectors_out = torch.stack(new_vectors_out, dim=0)  # [N, dim]

    # 4. Add noise (can be different per space to avoid identical init when untied)
    if add_noise and noise_scale > 0:
        noise_emb = torch.randn_like(new_vectors_emb) * noise_scale
        noise_out = torch.randn_like(new_vectors_out) * noise_scale
        new_vectors_emb = new_vectors_emb + noise_emb
        new_vectors_out = new_vectors_out + noise_out

    # 5. Align vector norms
    if align_norm_with_source_subtokens:
        aligned_emb_list = []
        aligned_out_list = []
        for i in range(len(target_tokens)):
            vec_emb = new_vectors_emb[i]
            vec_out = new_vectors_out[i]
            tgt_norm_emb = mean_source_norms_emb[i]
            tgt_norm_out = mean_source_norms_out[i]

            vec_emb = align_vector_norm(vec_emb, tgt_norm_emb)
            if not tied:
                vec_out = align_vector_norm(vec_out, tgt_norm_out)
            else:
                vec_out = vec_emb  # keep identical when tied

            aligned_emb_list.append(vec_emb)
            aligned_out_list.append(vec_out)

        new_vectors_emb = torch.stack(aligned_emb_list, dim=0)
        new_vectors_out = torch.stack(aligned_out_list, dim=0)

    # 6. Write back
    with torch.no_grad():
        for token_id, vec_emb, vec_out in zip(target_token_ids, new_vectors_emb, new_vectors_out):
            emb_weight[token_id] = vec_emb.to(emb_weight.dtype)
            if not tied:
                out_weight[token_id] = vec_out.to(out_weight.dtype)
            else:
                # No need to write out_weight when tied
                pass

    # 7. Save
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(f"[INFO] Saved to {save_dir}")

    # 8. Logging
    print("==== Init Summary ====")
    for idx, (t_tok, t_id, sub_ids) in enumerate(zip(target_tokens, target_token_ids, source_subtoken_id_lists)):
        final_emb_norm = emb_weight[t_id].norm().item()
        if not tied:
            final_out_norm = out_weight[t_id].norm().item()
        else:
            final_out_norm = final_emb_norm
        mean_emb_norm = mean_source_norms_emb[idx].item()
        mean_out_norm = mean_source_norms_out[idx].item() if not tied else mean_emb_norm

        print(
            f"Target {t_tok} (id={t_id}): "
            f"subtokens={sub_ids}, "
            f"mean_emb_source_norm={mean_emb_norm:.6f}, "
            f"mean_out_source_norm={mean_out_norm:.6f}, "
            f"final_emb_norm={final_emb_norm:.6f}, "
            f"final_out_norm={final_out_norm:.6f}"
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize special tokens for a model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models/Qwen/Qwen3-4B-Base",
        help="Path to the base model directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models/Qwen/Qwen3-4B-Base-inittag",
        help="Path to save the modified model and tokenizer.",
    )
    args = parser.parse_args()
    main(args.model_dir, args.save_dir)
