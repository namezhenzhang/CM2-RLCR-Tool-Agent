from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser(description='Download model from Hugging Face Hub')
parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B', help='Name of the model to download')
parser.add_argument('--local_dir', type=str, default='./', help='Local directory to download the model')
args = parser.parse_args()

local_dir = args.local_dir+"/"+args.model_name
snapshot_download(repo_id=args.model_name,
                  ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.gguf", "consolidated.safetensors"] ,
                  local_dir=local_dir)

# hf download namezz/cold-start-qwen-8b-base-inittag-keepthink-lr1e-5-gpu4-bs2-ga8-ep2-wr0.1-cut12000 --local-dir ./models/namezz/cold-start-qwen-8b-base-inittag-keepthink-lr1e-5-gpu4-bs2-ga8-ep2-wr0.1-cut12000

# hf download namezz/Checklist --local-dir ./data --token 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