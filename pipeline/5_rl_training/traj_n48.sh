set -x

PROJECT_DIR="path/to/CM2-RLCR-Tool-Agent/"
PROJECT_NAME="cm2-rlcr-tool-agent"
CONFIG_PATH="$PROJECT_DIR/pipeline/5_rl_training/config"
DATA_PATH="$PROJECT_DIR/data/rl_data"
REFERENCE_MODEL_PATH=$PROJECT_DIR/models/Qwen/cold-start-qwen-8b-base-inittag-keepthink-lr1e-5-gpu4-bs2-ga8-ep2-wr0.1-cut12000
LLM_AS_A_JUDGE_NAME="Qwen/Qwen3-4B-Instruct-2507"

NNODES=1
NUM_INFER_NODES=1
INFER_IP1='127.0.0.1'


# use algorithm.adv_estimator=grpo and +reward_model.reward_kwargs.reward_level=turn for trajetory level reward
REWARD_LEVEL="turn"
ROLL_OUT_N=4
LR=1e-6
NUM_GPUS_PER_NODE=4
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=8
TRAIN_BATCH_SIZE=4
MAX_PROMPT_LENGTH=4000
MAX_RESPONSE_LENGTH=8000
KL_LOSS_COEF=0.0001
TENSOR_MODEL_PARALLEL_SIZE=4
MAX_NUM_CHECKLIST=1

EXPERIMENT_NAME="drgrpo-tis-fix2-newdata-cold-start-qwen3-8b-base-notag-keepthink_bs-$TRAIN_BATCH_SIZE-n-$ROLL_OUT_N-c-$MAX_NUM_CHECKLIST_$MAX_PROMPT_LENGTH-${MAX_RESPONSE_LENGTH}_ppo-minibatch-$TRAIN_BATCH_SIZE-ppoepochs-1-kl-$KL_LOSS_COEF-ent-0.0-lr-$LR-node-$NNODES-$NUM_INFER_NODES-$LLM_AS_A_JUDGE_NAME"


export ENABLE_CHECKLIST=1


# uncomment to run on multiple nodes
# export RAY_ADDRESS='http://192.168.229.37:8265'
# ray job submit \
#     --runtime-env=$CONFIG_PATH/runtime_env.yaml \
#     -- \
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='checklist' \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_token_veto_threshold=1e-4 \
    algorithm.use_kl_in_reward=False \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.prompt_key=messages \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.custom_cls.path=pkg://verl.utils.dataset.checklist_dataset \
    data.custom_cls.name=ChecklistDataset \
    +data.max_num_checklist=$MAX_NUM_CHECKLIST \
    reward_model.reward_manager=checklist \
    +reward_model.reward_kwargs.sglang_url=[\
"http://${INFER_IP1}:10001/v1/chat/completions"\
] \
    +reward_model.reward_kwargs.sglang_model=$LLM_AS_A_JUDGE_NAME \
    +reward_model.reward_kwargs.retry_times=2 \
    +reward_model.reward_kwargs.semaphore_size=500 \
    +reward_model.reward_kwargs.timeout_seconds=1200 \
    +reward_model.reward_kwargs.temperature=0.6 \
    +reward_model.reward_kwargs.top_p=0.8 \
    +reward_model.reward_kwargs.max_new_tokens=6000 \
    +reward_model.reward_kwargs.max_tokens=6000 \
    +reward_model.reward_kwargs.eta=1.0 \
    +reward_model.reward_kwargs.reward_level=$REWARD_LEVEL \
    +reward_model.reward_kwargs.threthod_num=4 \
    +reward_model.reward_kwargs.do_norm_in_adv=False \
    actor_rollout_ref.model.path=$REFERENCE_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.skip_tokenizer_init=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.n=$ROLL_OUT_N \
    actor_rollout_ref.rollout.over_sample_rate=0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$PROJECT_DIR/models/verl_checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=$PROJECT_DIR/rollout_results/$PROJECT_NAME/$EXPERIMENT_NAME/train \
    trainer.validation_data_dir=$PROJECT_DIR/rollout_results/$PROJECT_NAME/$EXPERIMENT_NAME/valid \
    data.train_files=$DATA_PATH/rl_data_selected_checklist_annotated_fo_verl_train.parquet \
    data.val_files=$DATA_PATH/rl_data_selected_checklist_annotated_fo_verl_val.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/checklist_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_PATH/checklist_interaction_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.format=qwen25 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=30 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
    trainer.total_epochs=100 \
    actor_rollout_ref.nccl_timeout=900000 $@
