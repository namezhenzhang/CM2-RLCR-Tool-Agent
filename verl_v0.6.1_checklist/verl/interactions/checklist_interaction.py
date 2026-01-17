# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import logging
import os
import json
import asyncio
from typing import Any, Optional
from uuid import uuid4
import random

import httpx
from verl import DataProto
from verl.utils.reward_score import checklist_reward

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ChecklistInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of gsm8k.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self._sglang_url = config.get("sglang_url", [])
        self._sglang_model = config.get("sglang_model")
        self._retry_times = config.get("retry_times")
        self._semaphore_size = config.get("semaphore_size")
        self._temperature = config.get("temperature")
        self._top_p = config.get("top_p")
        self._max_new_tokens = config.get("max_new_tokens")
        self._max_tokens = config.get("max_tokens")
        self._timeout = config.get("timeout", 120.0)
        self._max_checklist_to_use = config.get("max_checklist_to_use",1)

        self._instance_dict = {}

        # Initialize shared client and semaphore
        try:
            self._timeout_value = float(self._timeout)
        except Exception:
            self._timeout_value = 120.0
        try:
            semaphore_size = int(self._semaphore_size) if self._semaphore_size is not None else 64
        except Exception:
            semaphore_size = 64
        self._semaphore = asyncio.Semaphore(semaphore_size)
        self._httpx_limits = httpx.Limits(
            max_connections=semaphore_size,
            max_keepalive_connections=semaphore_size // 2,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self._timeout_value,
                read=self._timeout_value,
                write=self._timeout_value,
                connect=self._timeout_value,
            ),
            limits=self._httpx_limits,
        )

        
    async def start_interaction(
        self, instance_id: Optional[str] = None, checklist_list: list[list[list[dict[str, Any]]]] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": 0.0,
            "turns": 0,
            "checklist_list": checklist_list,
            "num_turns": len(checklist_list[0])
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], all_messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        checklist_list = kwargs.get("checklist_list")
        results = []
        call_success = []
        for checklist in checklist_list:
            results.append(asyncio.create_task(self.generate_response_for_single_checklist(instance_id, messages, all_messages, checklist=checklist)))
            # await asyncio.sleep(0.001)
        results = await asyncio.gather(*results)

        per_step_results_list = [result[1] for result in results] # (len(checklist_list), step, len(this_turn_checklist)),  len(this_turn_checklist) is not the same for each checklist
        per_step_call_success_list = [result[2] for result in results]
        should_terminate_sequence_list = [result[0] for result in results]
        
        if self._instance_dict[instance_id]["turns"]+1 < self._instance_dict[instance_id]["num_turns"]:
            user_idx = 0
            for i in range(0, len(all_messages)):
                item = all_messages[i]
                if item.get("role") == "user":
                    if user_idx == self._instance_dict[instance_id]["turns"]+1:
                        response = item.get("content")
                        logger.debug(f"Proceeding with the next turn response: {response}")
                        break
                    user_idx += 1
            assert response != ""
            should_terminate_sequence = False
        else:
            response = ""
            should_terminate_sequence = True
        
        # Check if more than half of the should_terminate_sequence_list are True
        true_count = sum(should_terminate_sequence_list)
        if true_count > len(should_terminate_sequence_list) / 2:
            should_terminate_sequence = True
        
        for x in per_step_call_success_list:
            for y in x:
                for z in y:
                    if z==False:
                        logger.warning("reward gen failed, terminate seq")
                        should_terminate_sequence = True

        self._instance_dict[instance_id]["turns"] += 1

        return should_terminate_sequence, response, (per_step_results_list, [self._instance_dict[instance_id]["turns"]-1]*len(per_step_results_list[0]), per_step_call_success_list), {}
            
    async def generate_response_for_single_checklist(
        self, instance_id: str, messages: list[dict[str, Any]], all_messages: list[dict[str, Any]], checklist: list[list[dict[str, Any]]], **kwargs
    ) -> tuple[bool, str, float, dict]:


        this_turn_checklist = checklist[self._instance_dict[instance_id]["turns"]]
        not_required_for_next_turn_list = [not single_step_checklist["required_for_next_turn"] for single_step_checklist in this_turn_checklist]
        all_step_results_this_turn = []

        args = {
                "sglang_model": self._sglang_model,
                "sglang_url": None,
                "temperature": self._temperature,
                "top_p": self._top_p,
                "max_new_tokens": self._max_new_tokens,
                "max_tokens": self._max_tokens,
                "retry_times": self._retry_times
        }

        # find last user message idx
        last_user_message_idx = -1
        for i in range(len(messages)-1, -1, -1):
            if messages[i].role == "user":
                last_user_message_idx = i
                break
        assert last_user_message_idx != -1


        step = 0
        for i in range(last_user_message_idx+1, len(messages)):
            if messages[i].role == "assistant":
                this_step_message = [messages[i]]
                messages_before_this_step = messages[:i]
                this_step_message_str = checklist_reward.get_messages_str_v2(this_step_message, step)
                messages_str_before_this_turn = checklist_reward.get_messages_str_v2(messages[:last_user_message_idx])
                messages_str_before_this_step = checklist_reward.get_messages_str_v2(messages_before_this_step)
                following_tool_response_str = "No following tool response"
                this_turn_messages_util_now = messages[last_user_message_idx:i+1]
                tool_call_failed = False
                if i + 1 < len(messages) and messages[i + 1].role in ["observation", "tool"]:
                    tool_messages = []
                    j = i + 1
                    while j < len(messages) and messages[j].role in ["observation", "tool"]:
                        # Safely check for error_tool_call in message content
                        try:
                            content = messages[j].content
                            if content and isinstance(content, str):
                                parsed_content = json.loads(content)
                                if isinstance(parsed_content, dict) and "error_tool_call" in parsed_content:
                                    tool_call_failed = True
                        except (json.JSONDecodeError, TypeError):
                            # Content is not valid JSON or empty, skip error check
                            pass
                        tool_messages.append(messages[j])
                        this_turn_messages_util_now.append(messages[j])
                        j += 1
                    following_tool_response_str = checklist_reward.get_messages_str_v2(tool_messages)
                for single_step_checklist in this_turn_checklist:
                    # input_prompt = checklist_reward.get_input_prompt(messages_str_before_this_step, this_step_message_str, following_tool_response_str, single_step_checklist)
                    messages_str_in_this_turn_until_now = checklist_reward.get_messages_str_v2(this_turn_messages_util_now)
                    input_prompt = checklist_reward.get_input_prompt_v2(messages_str_before_this_turn, messages_str_in_this_turn_until_now, single_step_checklist)
                    selected_url = random.choice(self._sglang_url) if isinstance(self._sglang_url, list) and self._sglang_url else self._sglang_url
                    args["sglang_url"] = selected_url

                    async def _guarded_eval(prompt: str) -> bool:
                        async with self._semaphore:
                            return await checklist_reward.eval_one_check(self._client, prompt, args)
                    async def _guarded_eval_tool_error() -> bool:
                            return False, True  # type: ignore[arg-type]
                    if (single_step_checklist["focus_on"]=="assistant.tool_calls" or single_step_checklist["focus_on"]=="tool.content") and tool_call_failed:
                        all_step_results_this_turn.append(asyncio.create_task(_guarded_eval_tool_error()))
                    else:
                        all_step_results_this_turn.append(asyncio.create_task(_guarded_eval(input_prompt)))
                    # await asyncio.sleep(0.001)
                step += 1

        org_flat_per_step_results: list[(bool, bool)] = await asyncio.gather(*all_step_results_this_turn) # a list of bool lenght is steps * len(this_turn_checklist), the first len(this_turn_checklist) is for step 0

        assert len(org_flat_per_step_results) == len(this_turn_checklist) * (step), f"len(per_step_results) != len(this_turn_checklist) * step, {len(org_flat_per_step_results)} != {len(this_turn_checklist)} * {step}"

        flat_per_step_results = [x[0] for x in org_flat_per_step_results ]
        flat_per_step_call_success = [x[1] for x in org_flat_per_step_results ]


        per_step_results = [flat_per_step_results[i:i+len(this_turn_checklist)] for i in range(0, len(flat_per_step_results), len(this_turn_checklist))] # (step, len(this_turn_checklist))
        per_step_call_success = [flat_per_step_call_success[i:i+len(this_turn_checklist)] for i in range(0, len(flat_per_step_call_success), len(this_turn_checklist))] # (step, len(this_turn_checklist))


        turn = self._instance_dict[instance_id]["turns"]
        step = 0
        start = 0
        per_step_scores = []
        this_turn_checklist_mask = [1] * len(this_turn_checklist)
        for i in range(last_user_message_idx+1, len(messages)):
            message = messages[i]
            role = message.role
            assert role != "user"

            if role == "assistant":
                # calculate the score of this step
                # If one checklist is completed, later step can not finish it anymore
                # also check is all required_for_next_turn checklist are satisfied
                end = start + len(this_turn_checklist)
                this_step_results = flat_per_step_results[start:end]
                not_required_for_next_turn_list = [a or b for a,b in zip(not_required_for_next_turn_list, this_step_results)]
                weights = [float(single_step_checklist["weight"]) for single_step_checklist in this_turn_checklist]
                this_step_score = sum([weight * result * mask for weight, result, mask in zip(weights, this_step_results, this_turn_checklist_mask)])
                this_step_score = round(this_step_score, 8)
                this_turn_checklist_mask = [bool(int(a*(1-b))) for a,b in zip(this_turn_checklist_mask, this_step_results)]

                per_step_scores.append(this_step_score)
                start = end
                step += 1

        # reward = round(sum(per_step_scores), 4)

        if self._instance_dict[instance_id]["turns"]+1 < self._instance_dict[instance_id]["num_turns"]:
            user_idx = 0
            for i in range(0, len(all_messages)):
                item = all_messages[i]
                if item.get("role") == "user":
                    if user_idx == self._instance_dict[instance_id]["turns"]+1:
                        response = item.get("content")
                        logger.debug(f"Proceeding with the next turn response: {response}")
                        break
                    user_idx += 1
            assert response != ""
            should_terminate_sequence = False
        else:
            response = ""
            should_terminate_sequence = True
        
        if not all(not_required_for_next_turn_list):
            should_terminate_sequence = True
        



        return should_terminate_sequence, per_step_results, per_step_call_success

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

