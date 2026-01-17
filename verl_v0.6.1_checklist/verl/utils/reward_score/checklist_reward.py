import asyncio
import json
from collections import defaultdict
from typing import Any
import os
import re
import httpx
import torch
from verl import DataProto

import logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def extract_last_json(text, parse=True):
    FENCE_JSON_RE = re.compile(
        r"```(?:\s*json)?\s*(.*?)\s*```",
        re.IGNORECASE | re.DOTALL
    )
    matches = FENCE_JSON_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    if not parse:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None  # 解析失败就返回原始字符串


async def eval_one_check(client: httpx.AsyncClient, user_prompt: str, args: dict) -> bool:

    sglang_model = args.get("sglang_model")
    sglang_url = args.get("sglang_url")
    temperature = args.get("temperature")
    top_p = args.get("top_p")
    max_new_tokens = args.get("max_new_tokens")
    max_tokens = args.get("max_tokens")
    retry_times = args.get("retry_times")

    payload = {
        "model": sglang_model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_tokens": max_tokens,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        # "json_schema": {
        #     "type": "object",
        #     "properties": {
        #         "result": {"type": "boolean"}
        #     },
        #     "required": ["result"]
        # },
        # "response_format": {
        #     "type": "json_schema",
        #     "json_schema": {
        #         "name": "foo",
        #         "schema":{
        #             "type": "object",
        #             "properties": {
        #                 "result": {"type": "boolean"}
        #             },
        #             "required": ["result"]
        #         }
        #     }
        # }
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "title": "Checklist Evaluation Verdict",
                    "description": "Structured evaluation result for a single assistant turn against checklist criteria.",
                    "properties": {
                        "high_level_understanding_of_the_question": {
                            "type": "string",
                        },
                        "analysis_of_if_focus_on": {
                            "type": "string",
                        },
                        "analysis_of_pass_condition": {
                            "type": "string",
                        },
                        "analysis_of_failure_examples": {
                            "type": "string",
                        },
                        "answer": {
                            "type": "boolean",
                        }
                    },
                    "required": [
                        "high_level_understanding_of_the_question",
                        "analysis_of_if_focus_on",
                        "analysis_of_pass_condition",
                        "analysis_of_failure_examples",
                        "answer"
                    ],
                    "additionalProperties": False
                }
            }
        }
    }

    try:
        resp = await _post_with_retries(client, sglang_url, payload, retry_times)
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            # parsed = json.loads(text)
            parsed = json.loads(text)
            ans = parsed['answer']
            if ans not in [True, False]:
                raise ValueError("answer is not boolen")
            return ans, True
        except Exception as e:
            logger.warning(f"text can not be parsed in reward (call passed): {repr(e)}")
            return False, False
    except Exception as e:
        logger.warning(f"text can not be parsed in reward (call not passed): {repr(e)}")
        return False, False

def get_input_prompt(messages_str_before_this_step: str, this_step_message_str: str, following_tool_response_str: str, this_turn_checklist: list[dict[str, Any]]) -> str:
    reference_snippet = [evidence['snippet'] for evidence in this_turn_checklist['evidence']]
    input_prompt = (
        "# Instructions\n"
        "You are a strict checklist evaluator.\n"
        "You will be give a checklist for the assistant's new response.\n"
        "Checklist contains question, focus_on, pass_condition, failure_examples and reference snippet.\n"
        "Focus on is the part of the assistant's new response that the question is about.\n"
        "If the assistant's response follows the checklist's question, return true. Otherwise, return false.\n"
        "If the focus on is not in the assistant's new response or following tool response, return false.\n"
        "\n"
        "# Checklist:\n"
        f"Question: {this_turn_checklist['question']}\n"
        f"Focus on: {this_turn_checklist['focus_on']}\n"
        f"Pass condition: {this_turn_checklist['pass_condition']}\n"
        f"Failure examples: {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=False, indent=0)}\n"
        f"Reference snippet: {json.dumps(reference_snippet, ensure_ascii=False, indent=0)}\n"
        "\n"
        "# Previous messages:\n" + messages_str_before_this_step + "\n"
        "\n"
        "# Assistant's new response:\n" + this_step_message_str + "\n"
        "\n"
        "# Following tool response:\n" + following_tool_response_str + "\n"
        "# Response format:\n"
        "Return in JSON format: {'result': true/false}"
    )
    return input_prompt

def get_input_prompt_v2(messages_str_before_this_turn, messages_str_in_this_turn: str, this_turn_checklist: list[dict[str, Any]]) -> str:
    reference_snippet = [evidence['snippet'] for evidence in this_turn_checklist['evidence']]
    # input_prompt = (
    #     "# Instructions\n"
    #     "You are a strict checklist evaluator.\n"
    #     "You will be give messages between user, assistant and tools. And you will also be given a checklist for the assistant's response.\n"
    #     "Checklist contains question, focus_on, pass_condition, failure_examples and reference snippet.\n"
    #     "Focus on is the part of the assistant's response that the question is about.\n"
    #     "If the assistant's response follows the checklist's question and pass condition, return true. Otherwise, return false.\n"
    #     "If the focus on is not in the messages, return false.\n"
    #     "\n"
    #     "# Checklist:\n"
    #     f"Question: {this_turn_checklist['question']}\n"
    #     f"Focus on: {this_turn_checklist['focus_on']}\n"
    #     f"Pass condition: {this_turn_checklist['pass_condition']}\n"
    #     f"Failure examples: {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=False, indent=0)}\n"
    #     f"Reference snippet: {json.dumps(reference_snippet, ensure_ascii=False, indent=0)}\n"
    #     "\n"
    #     "# Messages:\n" + messages_str_in_this_turn + "\n"
    #     "# Response format:\n"
    #     "Return only in JSON format: {'result': true/false}"
    # )
    input_prompt = (
        "# Role\n"
        "You are a precise checklist evaluator. Your sole task is to judge whether the messages between user, assistant and tool satisfie the provided criteria.\n"
        "\n"
        "# Objective\n"
        "Produce a strict JSON verdict (no extra text) based on the instructions below.\n"
        "\n"
        "# Criteria\n"
        f"**Question:** {this_turn_checklist['question']}\n"
        f"**Focus on:** {this_turn_checklist['focus_on']}\n"
        f"**Pass condition:** {this_turn_checklist['pass_condition']}\n"
        f"**Failure examples:** {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=True, indent=2)}\n"
        f"**Reference snippet:** {json.dumps(reference_snippet, ensure_ascii=True, indent=2)}\n"
        "\n"
        "# Previous Messages\n"
        + messages_str_before_this_turn +
        "# Current Messages to Evaluate\n"
        + messages_str_in_this_turn +
        "\n"
        "# Special rule of tool call\n"
        "If there is no tool call in tool_call part but there are some tool calls in content.thinking part, it means these tools' format are not correct and all tool calls are not valid."
        "If there is error in tool response. The previous tool calls in latest assistant (only the latest one) are not valid."
        "# Evaluation Process (Align each step to a JSON output field)\n"
        "1. high_level_understanding_of_the_question:\n"
        "   - Briefly restate what is being evaluated (the intent of the question + what compliance means here).\n"
        "2. analysis_of_if_focus_on:\n"
        "   - Check whether Focus on part presents in the Current Messages.\n"
        "3. analysis_of_pass_condition:\n"
        "   - Determine if the 'Pass condition' is fully satisfied.\n"
        "4. analysis_of_failure_examples:\n"
        "   - For EACH failure example pattern: state clearly 'triggered' or 'not triggered' with a brief justification.\n"
        "5. answer:\n"
        "   - Return true ONLY IF:\n"
        "     * Focus on part is present.\n"
        "     * The 'Pass condition' is fully met.\n"
        "     * No failure example pattern is triggered.\n"
        "   - Otherwise return false.\n"
        "\n"
        "# Output Format\n"
        "Return ONLY a single JSON object with exactly these keys:\n"
        "{\n"
        "  \"high_level_understanding_of_the_question\": str,\n"
        "  \"analysis_of_if_focus_on\": str,\n"
        "  \"analysis_of_pass_condition\": str,\n"
        "  \"analysis_of_failure_examples\": str,\n"
        "  \"answer\": bool\n"
        "}"
    )

    # input_prompt = (
    #     "# Task\n"
    #     "You are a precise checklist evaluator that determines whether the messages between user, assistant and tool meet specific criteria.\n"
    #     "\n"
    #     "# Evaluation\n"
    #     "1. Locate the 'Focus on' content in the messages\n"
    #     "2. Check if the messages satisfies the 'Pass condition' of the question\n"
    #     "3. Compare against 'Failure examples' to avoid common mistakes\n"
    #     "4. Use 'Reference snippet' as a benchmark for expected quality if needed\n"
    #     "\n"
    #     "# Evaluation Criteria\n"
    #     f"**Question:** {this_turn_checklist['question']}\n"
    #     f"**Focus on:** {this_turn_checklist['focus_on']}\n"
    #     f"**Pass condition:** {this_turn_checklist['pass_condition']}\n"
    #     f"**Failure examples:** {json.dumps(this_turn_checklist['failure_examples'], ensure_ascii=True, indent=2)}\n"
    #     f"**Reference snippet:** {json.dumps(reference_snippet, ensure_ascii=True, indent=2)}\n"
    #     "\n"
    #     "# Decision Rules\n"
    #     "- Return `true` ONLY if:\n"
    #     "  * The 'Focus on' content is present in the assistant's response\n"
    #     "  * The response meets the 'Pass condition'\n"
    #     "  * The response avoids patterns shown in 'Failure examples'\n"
    #     "- Return `false` if:\n"
    #     "  * The 'Focus on' content is missing\n"
    #     "  * The 'Pass condition' is not satisfied\n"
    #     "  * The response matches any 'Failure examples'\n"
    #     "# Special criteria: Tool call format\n"
    #     "Each tool call made by the assistant must strictly adhere to the following format:\n"
    #     "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": { \"key\": \"value\" }}\n</tool_call>\n"
    #     "If a tool call deviates from this format, this tool call is not a valid tool call and should be considered incorrect.\n"
    #     "\n"
    #     "# Previous Messages\n"
    #     + messages_str_before_this_turn + 
    #     "# Messages to Evaluate\n"
    #     + messages_str_in_this_turn + 
    #     "\n\n# Output\n"
        
    #     "{\"high_level_understanding_of_the_question\": str,
    # "analysis_of_if_focus_on": str,
    # "analysis_of_pass_condition": str,
    # "analysis_of_failure_examples": str,
    # "answer": bool}"    
    
    return input_prompt

def get_messages_str_v1(messages: list[dict[str, Any]], step_num: int=None, max_length: int=40000) -> str:

    # TODO: limit the length of the messages_str
    if step_num is not None and messages[0].role == "assistant":
        assert len(messages) == 1, "Only one message is allowed when step_num is not None"
    turn = -1
    step = 0
    thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    messages_str = ""

    for i, message in enumerate(messages):
        role = message.role
        content = message.content
        if role == "assistant":
            if_thinking = thinking_regex.search(content)
            if if_thinking:
                thinking = if_thinking.group(1)
                user_visible_reply = content.split("</think>")[1].strip()
                if user_visible_reply == "":
                    user_visible_reply = "None"
            else:
                thinking = content
                user_visible_reply = "None"
            tool_calls = json.dumps([fm.model_dump() for fm in message.tool_calls] if message.tool_calls else [])
        
        if role == "system":
            messages_str += f"Role: system\ncontent: {content}\n"
            step = 0
        elif role == "user":
            turn += 1
            step = 0
            messages_str += f"# Turn: {turn}\nRole: user\ncontent: {content}\n"
        elif role == "assistant":
            if step_num is not None:
                _step = step_num
            else:
                _step = step
            this_step_message = f"## Step: {_step}\nRole: assistant\ncontent.thinking: {thinking}\ncontent.user_visible_reply: {user_visible_reply}\ntool_call: {tool_calls}\n"
            # for single_checklist in checklist[turn]:
            #     user_prompt = get_user_prompt_per_step(messages_before_this_step, this_step_message, single_checklist)
            #     all_step_results.append(eval_one_check(client, user_prompt, args))
            messages_str += this_step_message
            step += 1
        elif role == "observation" or role == "tool":
            messages_str += f"Role: tool\ncontent: {content}\n"
    return messages_str


def get_messages_str_v2(messages: list[dict[str, Any]], step_num: int=None, max_length: int=40000) -> str:

    # TODO: limit the length of the messages_str
    if step_num is not None and messages[0].role == "assistant":
        assert len(messages) == 1, "Only one message is allowed when step_num is not None"
    turn = -1
    step = 0
    thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    messages_str = ""

    for i, message in enumerate(messages):
        role = message.role
        content = message.content
        if role == "assistant":
            if_thinking = thinking_regex.search(content)
            if if_thinking:
                thinking = if_thinking.group(1)
                user_visible_reply = content.split(thinking+"</think>")[1].strip()
                if user_visible_reply == "":
                    user_visible_reply = "None"
            else:
                thinking = content
                user_visible_reply = "None"
            if message.tool_calls is not None:
                tool_calls = json.dumps([fm.model_dump() for fm in message.tool_calls] if message.tool_calls else [])
            elif "<tool_call>" in user_visible_reply or "</tool_call>" in user_visible_reply:
                tool_calls = user_visible_reply.replace("<tool_call>", "<|tool_call_start|>").replace("</tool_call>", "<|tool_call_end|>")
                user_visible_reply = "None"
            else:
                tool_calls = "None"
        
        if role == "system":
            messages_str += f"Role: system\ncontent: {content}\n"
            step = 0
        elif role == "user":
            turn += 1
            step = 0
            messages_str += f"# Turn: {turn}\nRole: user\ncontent: {content}\n"
        elif role == "assistant":
            if step_num is not None:
                _step = step_num
            else:
                _step = step
            this_step_message = f"## Step: {_step}\nRole: assistant\ncontent.thinking: {thinking}\ncontent.user_visible_reply: {user_visible_reply}\ntool_call: {tool_calls}\n"
            # for single_checklist in checklist[turn]:
            #     user_prompt = get_user_prompt_per_step(messages_before_this_step, this_step_message, single_checklist)
            #     all_step_results.append(eval_one_check(client, user_prompt, args))
            messages_str += this_step_message
            step += 1
        elif role == "observation" or role == "tool":
            messages_str += f"Role: tool\ncontent: {content}\n"
    return messages_str

# async def get_checklist_scores_per_step(messages_before_this_step: list[dict[str, Any]], this_step_message: str, this_turn_checklist: list[dict[str, Any]], args: dict) -> tuple[list[float], list[int]]:
#     turn = -1
#     step = 0
#     thinking_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)
#     tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

#     messages_before_this_step = ""
#     all_step_results = []
#     # semaphore = asyncio.Semaphore(self._semaphore_size)
#     async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=1200.0, read=1200.0, write=1200.0, connect=1200.0)) as client:
#         for i, message in enumerate(messages_before_this_step):
#             role = message.role
#             content = message.content
#             if role == "assistant":
#                 if_thinking = thinking_regex.search(content)
#                 if if_thinking:
#                     thinking = if_thinking.group(1)
#                     user_visible_reply = content.split("</think>")[1]
#                 else:
#                     thinking = content
#                     user_visible_reply = ""
#                 tool_calls = json.dumps([fm.model_dump() for fm in message.tool_calls] if message.tool_calls else [])
            
#             if role == "system":
#                 messages_before_this_step += f"Role: system\nContent: {content}\n"
#                 step = 0
#             elif role == "user":
#                 turn += 1
#                 step = 0
#                 messages_before_this_step += f"# Turn: {turn}\nRole: user\nContent: {content}\n"
#             elif role == "assistant":
#                 this_step_message = f"## Step: {step}\nRole: assistant\nThinking: {thinking}\nUser visible reply: {user_visible_reply}\nTool calls: {tool_calls}\n"
#                 for single_checklist in checklist[turn]:
#                     user_prompt = get_user_prompt_per_step(messages_before_this_step, this_step_message, single_checklist)
#                     all_step_results.append(eval_one_check(client, user_prompt, args))
#                 messages_before_this_step += this_step_message
#                 step += 1
#             elif role == "observation" or role == "tool":
#                 messages_before_this_step += f"Role: tool response\nContent: {content}\n"

    
#         all_step_results = await asyncio.gather(*all_step_results)

#     turn = -1
#     step = 0
#     start = 0
#     all_step_scores = []
#     turns = []
#     for i, message in enumerate(messages):
#         role = message.role
        
#         if role == "user":
#             step = 0
#             turn += 1
#             this_turn_checklist_mask = [1] * len(checklist[turn])
#         elif role == "assistant":
#             this_turn_checklist = checklist[turn]
#             end = start + len(this_turn_checklist)
#             this_step_results = all_step_results[start:end]
#             weights = [float(single_checklist["weight"]) for single_checklist in this_turn_checklist]
#             this_step_score = sum([weight * result * mask for weight, result, mask in zip(weights, this_step_results, this_turn_checklist_mask)])
#             this_turn_checklist_mask = [a*(1-b) for a,b in zip(this_turn_checklist_mask, this_step_results)]
#             this_step_score = round(this_step_score, 4)
#             all_step_scores.append(this_step_score)
#             turns.append(turn)
#             start = end
#             step += 1

#     return all_step_scores, turns



# def get_user_prompt_per_step(messages_before_this_step, this_step_message, this_turn_checklist):
#     user_prompt = (
#         "If the assistant's response follows the checklist, return True. Otherwise, return False.\n"
#         + "Checklist: " + json.dumps(this_turn_checklist, ensure_ascii=False, indent=0) + "\n"
#         + "Previous messages: " + json.dumps(messages_before_this_step, ensure_ascii=False, indent=0) + "\n"
#         + "Assistant's new response: " + json.dumps(this_step_message, ensure_ascii=False, indent=0) + "\n"
#         + "Return in JSON format: {'result': True/False}"
#     )
#     return user_prompt

async def get_checklist_scores_multiturn_multistep(
    messages,
    checklist,
    args: dict,
    calculated_rewards: list[float] | None = None,
    calculated_turns: list[int] | None = None,
    calculated_call_success: list[int] | None = None,
    client: httpx.AsyncClient | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[float], list[int]]:
    # Count the number of assistant messages
    assistant_count = sum(1 for message in messages if message.role == "assistant")
    if calculated_rewards is not None:
        if assistant_count == len(calculated_rewards):
            return calculated_rewards, calculated_turns, calculated_call_success

    turns = []
    all_step_scores = []
    call_success = []

    # if calculated_turns is not None:
    #     if len(calculated_turns) == 0:
    #         start_turn = 0
    #     else:
    #         start_turn = calculated_turns[-1]

    #     for i, calculated_turn in enumerate(calculated_turns):
    #         if calculated_turn < start_turn:
    #             all_step_scores.append(calculated_rewards[i])
    #             turns.append(calculated_turn)
    #         else:
    #             break

    if calculated_rewards is not None:
        if len(calculated_rewards) == 0:
            start_step = 0
        else:
            start_step = len(calculated_rewards)

        for i, calculated_reward in enumerate(calculated_rewards):
            all_step_scores.append(calculated_reward)
            turns.append(calculated_turns[i])
            call_success.append(calculated_call_success[i])

    turn = -1
    step = 0
    
    # Setup shared client and semaphore if not provided by caller
    # local_client: httpx.AsyncClient | None = None
    # if client is None:
    #     semaphore_size = int(args.get("semaphore_size", 64))
    #     timeout_seconds = float(args.get("timeout_seconds", 120.0))
    #     limits = httpx.Limits(
    #         max_connections=max(16, semaphore_size),
    #         max_keepalive_connections=max(8, semaphore_size // 2),
    #     )
    #     timeout = httpx.Timeout(timeout=timeout_seconds, read=timeout_seconds, write=timeout_seconds, connect=timeout_seconds)
    #     local_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    #     client = local_client
    # if semaphore is None:
    #     semaphore = asyncio.Semaphore(int(args.get("semaphore_size", 64)))

    all_step_results = []
    global_assistant_count = 0
    try:
        for i, message in enumerate(messages):
            role = message.role
            
            if role == "system":
                step = 0
            elif role == "user":
                turn += 1
                step = 0
            elif role == "assistant":
                if calculated_rewards is not None and start_step > global_assistant_count:
                        pass # already calculated
                else:
                    this_step_message = [messages[i]]
                    messages_before_this_step = messages[:i]
                    this_step_message_str = get_messages_str_v2(this_step_message, step)
                    messages_str_before_this_step = get_messages_str_v2(messages_before_this_step)
                    # Get following tool response if next message is a tool
                    following_tool_response_str = "No following tool response"
                    last_user_message_idx = -1
                    for i in range(len(messages)-1, -1, -1):
                        if messages[i].role == "user":
                            last_user_message_idx = i
                            break
                    assert last_user_message_idx != -1
                    messages_str_before_this_turn = get_messages_str_v2(messages[:last_user_message_idx])
                    this_turn_messages_util_now = messages[last_user_message_idx:i+1]
                    tool_call_failed = False
                    if i + 1 < len(messages) and messages[i + 1].role in ["observation", "tool"]:
                        tool_messages = []
                        j = i + 1
                        while j < len(messages) and messages[j].role in ["observation", "tool"]:
                            if "error_tool_call" in json.loads(messages[j].content):
                                tool_call_failed = True
                            tool_messages.append(messages[j])
                            this_turn_messages_util_now.append(messages[j])
                            j += 1
                        following_tool_response_str = get_messages_str_v2(tool_messages)
                    for single_step_checklist in checklist[turn]:
                        # input_prompt = get_input_prompt(messages_str_before_this_step, this_step_message_str, following_tool_response_str, single_step_checklist)
                        messages_str_in_this_turn_until_now = get_messages_str_v2(this_turn_messages_util_now)
                        input_prompt = get_input_prompt_v2(messages_str_before_this_turn, messages_str_in_this_turn_until_now, single_step_checklist)
                        async def _guarded_eval(prompt: str) -> bool:
                            async with semaphore:  # type: ignore[arg-type]
                                return await eval_one_check(client, prompt, args)  # type: ignore[arg-type]
                        async def _guarded_eval_tool_error() -> bool:
                            return False, True  # type: ignore[arg-type]
                        if (single_step_checklist["focus_on"]=="assistant.tool_calls" or single_step_checklist["focus_on"]=="tool.content") and tool_call_failed:
                            all_step_results.append(_guarded_eval_tool_error())
                        else:
                            all_step_results.append(_guarded_eval(input_prompt))
                step += 1
                global_assistant_count += 1
            elif role == "observation" or role == "tool":
                pass

        all_step_results = await asyncio.gather(*all_step_results)
    finally:
        # if local_client is not None:
        #     await local_client.aclose()
        pass


    turn = -1
    step = 0
    start = 0
    global_assistant_count = 0

    for i, message in enumerate(messages):
        role = message.role
        if role == "system":
            step = 0
        elif role == "user":
            step = 0
            turn += 1
            this_turn_checklist_mask = [1] * len(checklist[turn])
        elif role == "assistant":
            if calculated_rewards is not None and start_step > global_assistant_count:
                pass
            else:
                this_turn_checklist = checklist[turn]
                end = start + len(this_turn_checklist)
                this_step_results = all_step_results[start:end]
                all_step_scores.append([ x[0] for x in this_step_results])
                call_success.append([ x[1] for x in this_step_results])
                # weights = [float(single_checklist["weight"]) for single_checklist in this_turn_checklist]
                # this_step_score = sum([weight * result * mask for weight, result, mask in zip(weights, this_step_results, this_turn_checklist_mask)])
                # this_turn_checklist_mask = [a*(1-b) for a,b in zip(this_turn_checklist_mask, this_step_results)]
                # this_step_score = round(this_step_score, 8)
                # all_step_scores.append(this_step_score)
                turns.append(turn)
                start = end
            step += 1
            global_assistant_count += 1

    return all_step_scores, turns, call_success

async def _post_with_retries(client: httpx.AsyncClient, url: str, json_payload: dict, retry_times: int = 3) -> httpx.Response:
    """Post with retries and basic backoff. Caller should handle failures."""


    last_exc: Exception | None = None
    for attempt in range(max(1, int(retry_times))):
        try:
            resp = await client.post(url, json=json_payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # print(text,flush=True)
            parsed = json.loads(text)
            return resp
        except Exception as e:  # type: ignore[attr-defined]
            last_exc = e
            try:
                await asyncio.sleep(min((2 ** attempt)/10, 1))
            except Exception:
                pass
    # If all retries failed, re-raise the last exception
    assert last_exc is not None
    raise last_exc