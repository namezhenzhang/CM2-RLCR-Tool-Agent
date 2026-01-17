
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --tp-size 1 \
  --dp-size 4 \
  --context-length 20000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.70 \
  --port 10001
```

```bash
python ./run_rl/sglang_router_mcp_server_sse.py \
    --host 0.0.0.0 \
    --port 10002 \
    --dataset-path ./data/rl/checklist_annotated_rl_selected_by_stats_forverl.json \
    --sglang-url http://127.0.0.1:10001 \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --temperature 0.6 \
    --max-generated-tokens 2048 \
    --retry-attempts 1
```