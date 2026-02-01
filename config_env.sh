# 1. use docker
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/ --name verl verlai/verl:sgl055.latest sleep infinity
docker start verl
docker exec -it verl bash
cd /workspace
python3 -m pip install --upgrade pip
pip install mcp fastmcp

# 1. create conda environment
conda create -n verl python==3.11
conda activate verl

pip install --upgrade pip
pip install "numpy<2.0.0" # Fix numpy version conflict first (verl requires numpy<2.0.0)
pip install codetiming hydra-core peft pybind11 pylatexenc tensorboard tensordict torchdata wandb matplotlib
pip install torch transformers accelerate jsonlines math_verify openai torch_memory_saver sglang ray cachetools mcp fastmcp
pip install "flash_attn==2.6.3" --no-build-isolation


# 2. clone verl v0.6.1 and apply modifications
# If you directly use the verl_v0.6.1_checklist folder
cp -r verl_v0.6.1_checklist verl
cd verl
# If you want to use the latest verl (e.g., v0.7.0), you may need to resolve the version conflict, because the latest verl switches to Agentloop.
git clone git@github.com:volcengine/verl.git
cd verl
git checkout v0.6.1 # or any new version you want to use, e.g., v0.7.0
git apply --check ../verl_v0.6.1_modifications.diff
git apply ../verl_v0.6.1_modifications.diff

# 3. install verl in editable mode (now with dependencies satisfied)
pip3 install -e .[sglang]
cd ..

# 4. run the experiment
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rl/traj_n48.sh
