# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any
import os
import torch
import re
import math
import httpx
import asyncio
import json
import logging
import numpy as np
import random
import torch.distributed as dist

from verl import DataProto
from verl.utils.reward_score import checklist_reward, default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("checklist")
class ChecklistRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs: Any) -> None:
        """
        Initialize the ChecklistRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self._sglang_url = kwargs.get("sglang_url", [])
        self._sglang_model = kwargs.get("sglang_model", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self._retry_times = kwargs.get("retry_times", 3)
        self._semaphore_size = kwargs.get("semaphore_size", 20)
        self._timeout_seconds = kwargs.get("timeout_seconds", 120.0)
        self._temperature = kwargs.get("temperature", 0.6)
        self._top_p = kwargs.get("top_p", 0.8)
        self._max_new_tokens = kwargs.get("max_new_tokens", 100)
        self._max_tokens = kwargs.get("max_tokens", 2048)
        self._eta = kwargs.get("eta", 1.0)
        self._do_norm_in_adv = kwargs.get("do_norm_in_adv", False)
        self._reward_level = kwargs.get("reward_level", "turn")
        self._threthod_num = int(kwargs.get("threthod_num", 4))

        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


    def print_data(self, data: DataProto, i2scores, i2turn_counts):
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            # scores = scores_list[i]

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[checklists]", data.non_tensor_batch['extra_info'][i]['interaction_kwargs']['checklist_list'])
                print("[scores]", i2scores[i])
                print("[turn_counts]", i2turn_counts[i])
                # print("[scores]", scores)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Synchronous entrypoint that internally runs the async pipeline.

        Returns a torch.Tensor or a dict with keys 'reward_tensor' and 'reward_extra_info'.
        """
        
        result = asyncio.run(self._async_call(data, return_dict))
        return result

    async def _async_call(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:

        # data.batch = TensorDict(
        #     fields={
        #         attention_mask: Tensor(shape=torch.Size([32, 10000]), device=cpu, dtype=torch.int64, is_shared=False),
        #         input_ids: Tensor(shape=torch.Size([32, 10000]), device=cpu, dtype=torch.int64, is_shared=False),
        #         position_ids: Tensor(shape=torch.Size([32, 10000]), device=cpu, dtype=torch.int64, is_shared=False),
        #         prompts: Tensor(shape=torch.Size([32, 2000]), device=cpu, dtype=torch.int64, is_shared=False),
        #         response_mask: Tensor(shape=torch.Size([32, 8000]), device=cpu, dtype=torch.int64, is_shared=False),
        #         responses: Tensor(shape=torch.Size([32, 8000]), device=cpu, dtype=torch.int64, is_shared=False)},
        #     batch_size=torch.Size([32]),
        #     device=None,
        #     is_shared=False)

        logger.info("Getting checklist scores")
        # each element in scores_list[i] is a list of scores per assistant step for sample i
        def get_args_with_random_url():
            selected_url = random.choice(self._sglang_url) if self._sglang_url else None
            return {
                "sglang_model": self._sglang_model,
                "sglang_url": selected_url,
                "temperature": self._temperature,
                "top_p": self._top_p,
                "max_new_tokens": self._max_new_tokens,
                "max_tokens": self._max_tokens,
                "retry_times": self._retry_times,
            }

        semaphore_size = int(self._semaphore_size)
        limits = httpx.Limits(
            max_connections=semaphore_size,
            max_keepalive_connections=semaphore_size // 2,
        )
        timeout = httpx.Timeout(timeout=float(self._timeout_seconds), read=float(self._timeout_seconds), write=float(self._timeout_seconds), connect=float(self._timeout_seconds))
        semaphore = asyncio.Semaphore(semaphore_size)


        # tool_call_success = [
        #     all([
        #         metric["success"]
        #         for metrics_list in req_metrics.values()
        #         for metric in metrics_list
        #         if "success" in metric
        #     ])
        #     for req_metrics in data.non_tensor_batch["metrics"]
        # ]
        # logger.warning("{}/{} success".format(sum(tool_call_success), len(tool_call_success)))

        user_turn_rewards = [ x["user_turn_rewards"] for x in data.non_tensor_batch["reward_scores"]]

        # get reward that are not calculated in the last turn
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as shared_client:
            results = []
            for i in range(len(data)):
                this_data_checklist = data.non_tensor_batch['extra_info'][i]['interaction_kwargs']['checklist_list'] # 3 * turns * steps
                num_checklists = len(this_data_checklist)
                this_data_user_turn_rewards = user_turn_rewards[i] # turns * (reward, turn, call_success)
                this_data_rewards = [reward for reward, turn, success in this_data_user_turn_rewards] # turns * num_checklists * steps
                this_data_turns = [turn for reward, turn, success in this_data_user_turn_rewards] # turns * steps
                this_data_call_success = [success for reward, turn, success in this_data_user_turn_rewards] # turns * num_checklists * steps

                for j, this_checklist in enumerate(this_data_checklist):

                    this_checklist_this_data_rewards = [x[j] for x in this_data_rewards] # turns * steps
                    this_checklist_this_data_turns = this_data_turns # turns * steps
                    this_checklist_this_data_call_success = [x[j] for x in this_data_call_success] # turns * steps

                    flat_this_checklist_this_data_rewards = [item for sublist in this_checklist_this_data_rewards for item in sublist]
                    flat_this_checklist_this_data_turns = [item for sublist in this_checklist_this_data_turns for item in sublist]
                    flat_this_checklist_this_data_success = [item for sublist in this_checklist_this_data_call_success for item in sublist]

                    args = get_args_with_random_url()
                    # also pass semaphore size down for internal defaulting
                    args["semaphore_size"] = semaphore_size
                    args["timeout_seconds"] = float(self._timeout_seconds)
                    results.append(asyncio.create_task(checklist_reward.get_checklist_scores_multiturn_multistep(
                        data.non_tensor_batch["messages"][i]["messages"],
                        this_checklist,
                        args,
                        flat_this_checklist_this_data_rewards,
                        flat_this_checklist_this_data_turns,
                        flat_this_checklist_this_data_success,
                        client=shared_client,
                        semaphore=semaphore,
                    )))
                    # await asyncio.sleep(0.001)
            results = await asyncio.gather(*results)

        tool_call_success = [
            all([
                metric["success"]
                for metrics_list in req_metrics.values()
                for metric in metrics_list
                if "success" in metric
            ])
            for req_metrics in data.non_tensor_batch["metrics"]
        ]

        global_idx = 0
        for i in range(len(data)):
            this_data_checklist = data.non_tensor_batch['extra_info'][i]['interaction_kwargs']['checklist_list'] # 3 * turns * steps
            sample_base_idx = global_idx
            # this_data_result_list = []
            uuid = data.non_tensor_batch['extra_info'][i]["original_index"]
            should_block = False
            for j, this_checklist in enumerate(this_data_checklist):
                this_checklist_this_data_results = results[sample_base_idx + j]
                per_step_call_success = this_checklist_this_data_results[2] # list[list[bool]] per step
                for x in per_step_call_success:
                    for y in x:
                        if y == False:
                            should_block = True
                            break
                    if should_block:
                        break
            global_idx += len(this_data_checklist)
            if should_block:
                tool_call_success[i] = False

        logger.warning("{}/{} success (reward manager)".format(sum(tool_call_success), len(tool_call_success)))

        # merge all results to uuid2results
        global_idx = 0
        results_list = [] # len(data) * num_checklists * (scores_list, turns_list)
        uuid2results = {} # batchsize * num_checklist * (i, rewards, turns, dependence)
        uuid2idx = {} # uuid2idx[uuid] = [idx1, idx2, ...]
        for i in range(len(data)):
            this_data_checklist = data.non_tensor_batch['extra_info'][i]['interaction_kwargs']['checklist_list'] # 3 * turns * steps
            sample_base_idx = global_idx
            if not tool_call_success[i]:
                global_idx += len(this_data_checklist)
                # data.batch["response_mask"][i] = 0   
                continue
            # this_data_result_list = []
            uuid = data.non_tensor_batch['extra_info'][i]["original_index"]
            if uuid not in uuid2results or len(uuid2results[uuid])==0:
                uuid2results[uuid] = [[] for _ in range(len(this_data_checklist))]
            if uuid not in uuid2idx:
                uuid2idx[uuid] = []
            uuid2idx[uuid].append(i)
            for j, this_checklist in enumerate(this_data_checklist):
                this_checklist_this_data_results = results[sample_base_idx + j]
                all_turn_dependence_list = [self.get_dependence_per_turn_checklist(this_turn_this_checklist) for this_turn_this_checklist in this_checklist]
                # compute per-turn counts and aggregated standards
                per_step_standards = this_checklist_this_data_results[0]  # list[list[bool]] per step
                per_step_turn_indices = this_checklist_this_data_results[1]  # list[int] per step's turn index
                per_step_call_success = this_checklist_this_data_results[2] # list[list[bool]] per step
                num_turns = len(this_checklist)
                per_turn_counts = [0] * num_turns
                per_turn_aggregated_standards = [[False] * len(this_checklist[t]) for t in range(num_turns)]
                per_turn_weights = [[y['weight']for y in x] for x in this_checklist]
                for step_standards, turn_idx in zip(per_step_standards, per_step_turn_indices):
                    if 0 <= turn_idx < num_turns:
                        per_turn_counts[turn_idx] += 1
                        assert len(step_standards) == len(per_turn_aggregated_standards[turn_idx]), f"{len(step_standards)}, {len(per_turn_aggregated_standards[turn_idx])}"
                        for k in range(len(step_standards)):
                            if step_standards[k]:
                                per_turn_aggregated_standards[turn_idx][k] = True
                # compute weighted per-turn scores: sum(weight_k * standard_k)
                per_turn_scores = []
                for t in range(num_turns):
                    if per_turn_counts[t] == 0:
                        per_turn_scores.append(0.0)
                        continue
                    standards_t = per_turn_aggregated_standards[t]
                    # already normalized to 1.0
                    weights_t = per_turn_weights[t]
                    numer = 0.0
                    for s_val, w_val in zip(standards_t, weights_t):
                        numer += (1.0 if bool(s_val) else 0.0) * float(w_val)
                    per_turn_scores.append(numer)

                uuid2results[uuid][j].append((i, this_checklist_this_data_results[0], this_checklist_this_data_results[1], all_turn_dependence_list, per_turn_counts, per_turn_scores, per_turn_aggregated_standards, per_turn_weights)) # batchsize * num_checklist * n * (succes_turn, success_turn, all_turn, per_turn_counts, per_turn_aggregated, per_turn_weights, per_turn_scores)
            global_idx += len(this_data_checklist)

        # implement logic.txt: compute effective ranges, pass rates and per-step rewards
        # 将二进制mask中连续的1段转换为[(start, end)]区间（end为开区间）
        # 用于把assistant的每段回复映射到对应的末token位置
        def one_segments_torch(mask):
            m = mask.to(torch.int8) if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.int8)
            diff = torch.diff(torch.cat([torch.tensor([0], dtype=torch.int8), m, torch.tensor([0], dtype=torch.int8)]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]
            return list(zip(starts.tolist(), ends.tolist()))
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        turn_end_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.bool)
        adv_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        i2scores = [None]*len(data)
        i2turn_counts = [None]*len(data)
        for uuid, all_checklist_results in uuid2results.items():
            for this_checklist_results in all_checklist_results:
                for this_checklist_result in this_checklist_results:
                    i = this_checklist_result[0]
                    per_turn_counts = this_checklist_result[4]
                    per_turn_scores = this_checklist_result[5]
                    i2scores[i] = per_turn_scores
                    i2turn_counts[i] = per_turn_counts
                    accumulate_steps = 0
                    segments = one_segments_torch(data.batch["response_mask"][i])
                    for count, score in zip(per_turn_counts, per_turn_scores):
                        if count == 0:
                            break
                        if accumulate_steps+count-1 >= len(segments):
                            break
                        end = segments[accumulate_steps+count-1][1]
                        reward_tensor[i, end-1] += float(score)
                        turn_end_tensor[i, end-1] = True
                        accumulate_steps += count
        self.print_data(data,i2scores,i2turn_counts)
        # norm by num of checklists
        num_checklists = torch.tensor(
            [len(data.non_tensor_batch['extra_info'][i]['interaction_kwargs']['checklist_list']) for i in range(len(data))],
            device=reward_tensor.device,
            dtype=reward_tensor.dtype,
        )
        reward_tensor /= num_checklists.unsqueeze(1)

        # norm by num of max turns
        max_num_turns_tensor = torch.tensor(
            [len(data.non_tensor_batch['extra_info'][i]['interaction_kwargs']['checklist_list'][0]) for i in range(len(data))],
            device=reward_tensor.device,
            dtype=reward_tensor.dtype,
        )
        reward_tensor /= max_num_turns_tensor.unsqueeze(1)

        num_turns_tensor = torch.tensor(
            [sum(1 for m in data.non_tensor_batch["messages"][i]["messages"] if m.role=="user") for i in range(len(data))],
            device=reward_tensor.device,
            dtype=reward_tensor.dtype,
        )

        tool_call_success = torch.tensor(
            tool_call_success,
            device=reward_tensor.device,
            dtype=torch.bool,
        )
        # reward_tensor *= tool_call_success.unsqueeze(1)

        if self._reward_level == "step":
            logger.warning("calculate step level reward")
            # 步骤1：计算每个uuid的每个checklist的每个turn的每条标准的平均值、标准差
            uuid_checklist_turn_standard_stats = {}  # uuid -> checklist_idx -> turn_idx -> standard_idx -> (mean, std)

            for uuid, all_checklist_results in uuid2results.items():
                uuid_checklist_turn_standard_stats[uuid] = {}

                for checklist_idx, this_checklist_results in enumerate(all_checklist_results):
                    if len(this_checklist_results) == 0:
                        continue

                    # 获取依赖关系（所有sample的依赖关系相同）
                    dependence_list = this_checklist_results[0][3]
                    num_turns = len(dependence_list)

                    uuid_checklist_turn_standard_stats[uuid][checklist_idx] = {}

                    # 收集每个turn的每条标准的所有有效分数
                    turn_standard_scores = {}  # turn_idx -> standard_idx -> list of scores

                    for this_checklist_result_idx, this_checklist_result in enumerate(this_checklist_results):
                        step_standards = this_checklist_result[1]  # list[list[bool]] per step
                        step_turn_indices = this_checklist_result[2]  # list[int] per step's turn index

                        # 对每个turn，跟踪已完成的标准
                        turn_completed_standards = {}  # turn_idx -> set of completed standard_idx

                        # 按step顺序处理，跟踪动态依赖关系
                        previous_turn = -1
                        fist_step_this_turn = -1
                        for step_idx, (step_standard_results, turn_idx) in enumerate(zip(step_standards, step_turn_indices)):
                            # if turn_idx >= num_turns or turn_idx < 0:
                            #     continue
                            assert 0 <= turn_idx < num_turns
                            if turn_idx!=previous_turn:
                                previous_turn = turn_idx
                                fist_step_this_turn = step_idx

                            if turn_idx not in turn_completed_standards:
                                turn_completed_standards[turn_idx] = {}

                            turn_dependence = dependence_list[turn_idx]
                            for standard_idx, standard_result in enumerate(step_standard_results):
                                if standard_idx >= len(turn_dependence):
                                    continue

                                # 检查该标准的依赖是否在这个turn中已经被前面的step完成了
                                dependencies_met = True
                                max_step_dependencies_met = -1
                                if len(turn_dependence[standard_idx])==0:
                                    max_step_dependencies_met = fist_step_this_turn-1
                                for dep_idx in turn_dependence[standard_idx]:
                                    if dep_idx not in turn_completed_standards[turn_idx]:
                                        dependencies_met = False
                                        break
                                    else:
                                        max_step_dependencies_met = max(max_step_dependencies_met, turn_completed_standards[turn_idx][dep_idx])
                                    

                                # 如果依赖满足且该标准在此turn中尚未完成，收集该标准的分数
                                if dependencies_met and standard_idx not in turn_completed_standards[turn_idx]:
                                    if turn_idx not in turn_standard_scores:
                                        turn_standard_scores[turn_idx] = {}
                                    if standard_idx not in turn_standard_scores[turn_idx]:
                                        turn_standard_scores[turn_idx][standard_idx] = [np.nan] * len(this_checklist_results)
                                    if standard_result:
                                        standard_score = 1.0 * (self._eta**(step_idx-max_step_dependencies_met-1))
                                        turn_standard_scores[turn_idx][standard_idx][this_checklist_result_idx] = standard_score
                                    else:
                                        if np.isnan(turn_standard_scores[turn_idx][standard_idx][this_checklist_result_idx]):
                                            turn_standard_scores[turn_idx][standard_idx][this_checklist_result_idx] = 0.0

                                    # 如果标准完成了且依赖项都已完成，才标记为已完成
                                    if standard_result:
                                        # turn_completed_standards[turn_idx].add(standard_idx)
                                        turn_completed_standards[turn_idx][standard_idx] = step_idx

                    # 计算每个turn每条标准的平均值和标准差
                    for turn_idx in range(num_turns):
                        uuid_checklist_turn_standard_stats[uuid][checklist_idx][turn_idx] = {}

                        if turn_idx in turn_standard_scores:
                            for standard_idx, scores in turn_standard_scores[turn_idx].items():
                                not_nan_score = [ x for x in scores if not np.isnan(x)]
                                if len(not_nan_score) > 0:
                                    mean_score = np.mean(not_nan_score)
                                    std_score = np.std(not_nan_score) if len(not_nan_score) > 1 else 1.0  # 避免一个元素报错
                                    std_score = max(std_score, 1e-6)  # 避免标准差为0
                                    uuid_checklist_turn_standard_stats[uuid][checklist_idx][turn_idx][standard_idx] = (mean_score, std_score, len(not_nan_score))

            all_std_num = 0
            all_std_num_lt_4 = 0
            threthod_num = self._threthod_num
            # 步骤2和3：计算每个数据的每个step的advantage并分配到tokens
            for uuid, all_checklist_results in uuid2results.items():
                # 按数据索引分组，将同一数据的所有checklist的结果组合在一起
                data_idx_to_checklist_results = {}
                for checklist_idx, this_checklist_results in enumerate(all_checklist_results):
                    for this_checklist_result in this_checklist_results:
                        i = this_checklist_result[0]  # 数据索引
                        if i not in data_idx_to_checklist_results:
                            data_idx_to_checklist_results[i] = []
                        data_idx_to_checklist_results[i].append((checklist_idx, this_checklist_result))

                # 对每个数据计算advantage
                for i, checklist_results in data_idx_to_checklist_results.items():
                    # 获取response的segments
                    segments = one_segments_torch(data.batch["response_mask"][i])

                    # 计算总共的step数量
                    total_steps = len(checklist_results[0][1][1])  # 取第一个checklist的step数量

                    # 对每个step计算advantage
                    for step_idx in range(total_steps):
                        if step_idx >= len(segments):
                            break

                        # 收集所有checklist在这个step的标准分数、均值、标准差
                        all_actual_scores = []
                        all_mean_scores = []
                        all_std_scores = []
                        all_weights = []

                        for checklist_idx, this_checklist_result in checklist_results:
                            step_standards = this_checklist_result[1]
                            step_turn_indices = this_checklist_result[2]
                            dependence_list = this_checklist_result[3]
                            per_turn_weights = this_checklist_result[7]  # 获取权重信息

                            if step_idx >= len(step_standards):
                                continue

                            step_standard_results = step_standards[step_idx]
                            turn_idx = step_turn_indices[step_idx]

                            if turn_idx < 0 or turn_idx >= len(dependence_list):
                                continue

                            # 检查该step在统计数据中是否存在
                            if (uuid not in uuid_checklist_turn_standard_stats or 
                                checklist_idx not in uuid_checklist_turn_standard_stats[uuid] or
                                turn_idx not in uuid_checklist_turn_standard_stats[uuid][checklist_idx]):
                                continue
                            
                            turn_stats = uuid_checklist_turn_standard_stats[uuid][checklist_idx][turn_idx]
                            turn_dependence = dependence_list[turn_idx]
                            turn_weights = per_turn_weights[turn_idx]
                            
                            # 跟踪这个turn中已完成的标准（重新模拟前面的步骤）
                            turn_completed_standards = set()
                            for prev_step_idx in range(step_idx):
                                if prev_step_idx >= len(step_standards):
                                    break
                                prev_step_results = step_standards[prev_step_idx]
                                prev_turn_idx = step_turn_indices[prev_step_idx]
                                if prev_turn_idx == turn_idx:
                                    for std_idx, std_result in enumerate(prev_step_results):
                                        if std_result and std_idx < len(turn_dependence):
                                            # 检查该标准的依赖是否都已完成
                                            dependencies_met = True
                                            for dep_idx in turn_dependence[std_idx]:
                                                if dep_idx not in turn_completed_standards:
                                                    dependencies_met = False
                                                    break
                                            # 只有当标准完成且其依赖项也都完成时才标记为已完成
                                            if dependencies_met:
                                                turn_completed_standards.add(std_idx)
                            
                            # 检查整个turn中该标准是否被完成（包括当前step之后的step），需要考虑依赖关系
                            # turn_standard_completion = {}  # standard_idx -> bool
                            turn_completed_for_full_check = {}  # 用于完整turn检查的已完成标准集合
                            
                            # 按step顺序重新模拟整个turn，考虑依赖关系
                            first_step_this_turn = -1
                            previous_turn = -1
                            for check_step_idx in range(len(step_standards)):
                                check_turn_idx = step_turn_indices[check_step_idx]
                                if check_turn_idx != previous_turn:
                                    previous_turn = check_turn_idx
                                    first_step_this_turn = check_step_idx
                                if check_turn_idx == turn_idx:
                                    check_step_results = step_standards[check_step_idx]
                                    for std_idx, std_result in enumerate(check_step_results):
                                        if std_result:
                                            # 检查该标准的依赖是否都已完成
                                            dependencies_met = True
                                            max_step_dependencies_met = -1
                                            if len(turn_dependence[std_idx])==0:
                                                max_step_dependencies_met = first_step_this_turn-1
                                            for dep_idx in turn_dependence[std_idx]:
                                                if dep_idx not in turn_completed_for_full_check:
                                                    dependencies_met = False
                                                    break
                                                else:
                                                    max_step_dependencies_met = max(max_step_dependencies_met, turn_completed_for_full_check[dep_idx])

                                            # 只有当标准完成且依赖项都满足时，才标记为真正完成
                                            if dependencies_met:
                                                # turn_standard_completion[std_idx] = True
                                                turn_completed_for_full_check[std_idx]=check_step_idx-max_step_dependencies_met-1
                            
                            # 检查当前step中每个标准的依赖和有效性
                            for standard_idx, standard_result in enumerate(step_standard_results):
                                if (standard_idx >= len(turn_dependence) or 
                                    standard_idx not in turn_stats):
                                    continue
                                
                                # 检查该标准的依赖是否在这个turn中已经被前面的step完成了
                                dependencies_met = True
                                for dep_idx in turn_dependence[standard_idx]:
                                    if dep_idx not in turn_completed_standards:
                                        dependencies_met = False
                                        break
                                
                                # 如果依赖满足，且该标准在此turn中尚未完成，收集该标准的分数
                                if dependencies_met and standard_idx not in turn_completed_standards:
                                    mean_score, std_score, num_scores = turn_stats[standard_idx]
                                    # 修改判断逻辑：如果依赖项都完成了，之前没完成，并且这个turn内完成了这个标准（可能在后面完成），都算对
                                    all_std_num += 1
                                    if num_scores > threthod_num:
                                        actual_score = 1.0 * (self._eta ** turn_completed_for_full_check[standard_idx]) if standard_idx in turn_completed_for_full_check else 0.0
                                    else:
                                        all_std_num_lt_4 += 1
                                        actual_score = mean_score

                                    # 获取该标准的权重
                                    weight = turn_weights[standard_idx]
                                    
                                    # 按权重加权收集分数
                                    all_actual_scores.append(actual_score * weight)
                                    all_mean_scores.append(mean_score * weight)
                                    all_std_scores.append(std_score * weight)
                                    all_weights.append(weight)
                        # 计算该step的advantage：所有标准分数之和 - 所有均值之和，除以所有标准差的几何平均值
                        if len(all_actual_scores) > 0:
                            total_actual = sum(all_actual_scores)
                            total_mean = sum(all_mean_scores)
                            total_weight = sum(all_weights)
                            total_weight = total_weight if total_weight != 0 else 1.0
                            geometric_mean_std = np.sqrt(np.sum(np.square(all_std_scores)))  # 平方求和再开根号
                            geometric_mean_std = geometric_mean_std if geometric_mean_std != 0 else 1.0
                            
                            # final_advantage = (total_actual - total_mean) / (geometric_mean_std if self._do_norm_in_adv else total_weight)
                            final_advantage = (total_actual - total_mean) / (total_weight if self._do_norm_in_adv else 1.0)
                            
                            # 将advantage分配到该step的所有tokens
                            segment_start, segment_end = segments[step_idx]
                            adv_tensor[i, segment_start:segment_end] = final_advantage
            if all_std_num==0:
                all_std_num=1
            logger.warning(f"step adv: {all_std_num_lt_4}/{all_std_num} ({all_std_num_lt_4/all_std_num*100}%) is ignored because group <= {threthod_num}")
        if return_dict:
            if self._reward_level == "step":
                reward_extra_info = {
                        "adv_tensor": adv_tensor,
                        "turn_end_tensor": turn_end_tensor,
                        "max_num_turns": max_num_turns_tensor,
                        "num_turns": num_turns_tensor,
                        "tool_call_success": tool_call_success
                    }
            else:
                reward_extra_info = {
                    "turn_end_tensor": turn_end_tensor,
                    "max_num_turns": max_num_turns_tensor,
                    "num_turns": num_turns_tensor,
                    "tool_call_success": tool_call_success
                }
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor

    
    def get_dependence_per_turn_checklist(self, this_turn_checklist: list[dict[str, Any]]) -> list[dict[str, Any]]:
        id2idex = {item["id"]: idx for idx, item in enumerate(this_turn_checklist)}
        dependence_list = []
        for idx, item in enumerate(this_turn_checklist):
            dependence = item["dependence"]
            dependence_list.append([id2idex[dep] for dep in dependence])
        return dependence_list
