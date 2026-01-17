# work dir: pipeline

# 1. Data Downloading
# Login using e.g. `huggingface-cli login` to access this dataset
mkdir -p ../data/
hf download nvidia/Nemotron-Post-Training-Dataset-v1 \
  --repo-type dataset \
  --include "data/tool-*" ".gitattributes" "README.md" \
  --local-dir ../data/nemotron_post_training_dataset_v1/tool_calling


# 2. Data Filtering

# filter data by rule-based filtering
python 1_data_filtering/rule_based_data_filtering.py \
  --n_samples -1 \
  --tool_schema \
  --role_order \
  --tool_call_response_match \
  --no_tool_response_in_assistant_message \
  --tool_response_json_valid \
  --duplicate_tools_by_name \
  --non_empty_content \
  --think_tags \
  --dataset_dir ../data/nemotron_post_training_dataset_v1/tool_calling \
  --output_dir ../data/rule_based_filtered_data

# error type (have overlap): {'tool_schema': 9195, 'role_order': 641, 'tool_call_response_match': 776, 'no_tool_response_in_assistant_message': 63, 'tool_response_json_valid': 4486, 'duplicate_tools_by_schema': 0, 'duplicate_tools_by_name': 20893, 'non_empty_content': 3111, 'think_tags': 649}
# Filtered 272890 samples

# compute statistic for rule-based filtered data
python utils/data_distribution.py \
    --dataset_dir ../data/rule_based_filtered_data \
    --output ../data/rule_based_filtered_data/data_distribution.jsonl \
    --n_samples -1 \
    --plot_metric all \
    --plot_output ../data/rule_based_filtered_data/plots \
    --plot_bins -1


# llm-based filtering
# We use gpt-5 with different effort to filter data with logic error.
# The effort is set to low*2, medium*2, and high*4.
export OPENAI_API_KEY=<OPENAI_API_KEY>

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/rule_based_filtered_data \
  --output ../data/llm_based_filtered_data/llm_based_filtered_low1.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_low1_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort low \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt \
  # for testing
  # --n-samples 20 \
  # --seed 42

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_low1.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_low2.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_low2_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort low \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_low2.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_medium1.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_medium1_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort medium \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt


OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_medium1.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_medium2.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_medium2_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort medium \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_medium2.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_high1.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_high1_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort high \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_high1.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_high2.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_high2_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort high \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_high2.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_high3.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_high3_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort high \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 1_data_filtering/llm_based_data_filtering.py \
  --input ../data/llm_based_filtered_data/llm_based_filtered_high3.json \
  --output ../data/llm_based_filtered_data/llm_based_filtered_high4.json \
  --raw-output ../data/llm_based_filtered_data/llm_based_filtered_high4_raw.json \
  --model gpt-5 \
  --workers 1000 \
  --effort high \
  --save-every 1000 \
  --max-retries 3 \
  --system-prompt-file prompts/llm_based_data_filtering_system_prompt.txt

cp ../data/llm_based_filtered_data/llm_based_filtered_high4.json ../data/llm_based_filtered_data/llm_based_filtered_final.json

# compute statistic for llm-based filtered data
python utils/data_distribution.py \
    --dataset_dir ../data/llm_based_filtered_data/llm_based_filtered_final.json \
    --output ../data/llm_based_filtered_data/llm_based_filtered_final_data_distribution.jsonl \
    --n_samples -1 \
    --plot_metric all \
    --plot_output ../data/llm_based_filtered_data/llm_based_filtered_final_plots \
    --plot_bins -1



# 3. CoT Compression

# Split 8k Cold Start 
python 2_cot_compression/split_cold_start_data.py \
  --n_samples 8000 \
  --input ../data/llm_based_filtered_data/llm_based_filtered_final.json \
  --rl_output ../data/rl_data/rl_data.json \
  --cold_start_output ../data/cold_start_data/cold_start_data.json

# statistic for cold start data
python utils/data_distribution.py \
    --dataset_dir ../data/cold_start_data/cold_start_data.json \
    --output ../data/cold_start_data/cold_start_data_data_distribution.jsonl \
    --n_samples -1 \
    --plot_metric all \
    --plot_output ../data/cold_start_data/cold_start_data_plots \
    --plot_bins -1

# Compression
OPENAI_API_KEY=<OPENAI_API_KEY> \
python 2_cot_compression/cot_compression_v1.py \
  --input-json ../data/cold_start_data/cold_start_data.json \
  --output-file ../data/cold_start_data/cold_start_data_cot_compressed.json \
  --prompt-file prompts/cot_compression_prompt_v1.txt \
  --n_samples -1 \
  --model gpt-5 \
  --effort medium \
  --verbosity medium \
  --num_workers 1000 \
  --retry 3 \
  --save_every 1000 \
  --save_every_seconds 120 \
  --seed 42

# 4. Cold Start SFT

# convert to llamafactory format
# modify dataset_info.json to support the new format or dataset
python 3_cold_start_sft/convert_cold_start_for_llamafactory.py \
  --input ../data/cold_start_data/cold_start_data_cot_compressed.json \
  --output ../data/cold_start_data/cold_start_data_cot_compressed_for_llamafactory.json

# copy dataset_info.json to support the new format or dataset
cp 3_cold_start_sft/dataset_info.json ../data/cold_start_data/dataset_info.json

# download model
hf download Qwen/Qwen3-8B-Base \
  --local-dir ../models/Qwen/Qwen3-8B-Base

# initialize special tokens
python ../model_assets/init_tag.py \
    --model_dir ../models/Qwen/Qwen3-8B-Base \
    --save_dir ../models/Qwen/Qwen3-8B-Base-inittag

# llamafactory training
WANDB_API_KEY=<WANDB_API_KEY> \
llamafactory-cli train run_cold_start/cold_start_qwen3-8b-base-inittag-keepthink.yaml

# copy and replace the chat template for verl training
cp ../model_assets/chat_template.json path/to/cold_start_model/chat_template.json


# 5. Checklist Labeling

# filtering by stats to get harder rl data
python utils/filter_by_stats.py \
  --dataset_dir ../data/rl_data/rl_data.json \
  --output_dir ../data/rl_data/rl_data_selected.json \
  --seed 42

# statistic for rl data
python utils/data_distribution.py \
    --dataset_dir ../data/rl_data/rl_data_selected.json \
    --output ../data/rl_data/rl_data_selected_data_distribution.jsonl \
    --n_samples -1 \
    --plot_metric all \
    --plot_output ../data/rl_data/rl_data_selected_plots \
    --plot_bins -1

OPENAI_API_KEY=<OPENAI_API_KEY> \
python 4_checklist_labeling/checklist_labeling_v3.py \
  --dataset_dir ../data/rl_data/rl_data_selected.json \
  --output_file ../data/rl_data/rl_data_selected_checklist_annotated.json \
  --prompt_file prompts/checklist_labeling_prompt_v3.txt \
  --n_samples -1 \
  --num_checklists 1 \
  --model gpt-5 \
  --effort high \
  --num_workers 1000 \
  --retry 3 \
  --save_every 1000 \
  --save_every_seconds 120 \
  --seed 42


# 6. RL Training

# copy and replace the chat template for verl training
cp ../model_assets/chat_template.json path/to/cold_start_model/chat_template.json

# Convert data format for verl training
python 5_rl_training/convert_checklist_for_verl.py \
  --input_path ../data/rl_data/rl_data_selected_checklist_annotated.json \
  --output_path ../data/rl_data/rl_data_selected_checklist_annotated_for_verl.json

# Convert to parquet for verl training
python 5_rl_training/convert_checklist_to_parquet.py \
  --input ../data/rl_data/rl_data_selected_checklist_annotated_for_verl.json \
  --output ../data/rl_data/rl_data_selected_checklist_annotated_fo_verl.parquet \
  --parse-tools-json \
  --data-source nvidia_nemotron_checklist \
  --n-val 500

# Start sglang server for llm-as-a-judge and tool simulation
# We suggest increasing the dp-size for better performance if you have more GPUs and memory. Otherwise, increase the tp-size.
# Use at least 1:1 gpu ratio of rl training for better performance
CUDA_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --tp-size 1 \
  --dp-size 1 \
  --context-length 20000 \
  --mem-fraction-static 0.70 \
  --host 0.0.0.0 \
  --port 10001


# Start sglang router mcp server for tool simulation
python 5_rl_training/sglang_router_mcp_server_sse.py \
    --host 0.0.0.0 \
    --port 10002 \
    --dataset-path ../data/rl_data/rl_data_selected_checklist_annotated_for_verl.json

# RL Training
# Adjust the config file (e.g., path, hyperparameters) and run the script
bash 5_rl_training/traj_n48.sh

# Model Merge
# Adjust the model path to merge and save the merged model after RL training
python ../model_assets/model_merge.py \
  --checkpoint_dir model_path_to_merge \
  --output_dir model_path_to_save_merged

