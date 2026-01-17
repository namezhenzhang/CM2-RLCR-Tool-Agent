# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple
import httpx
from collections import defaultdict
from mcp.types import ContentBlock, Tool as MCPTool
import threading
import pickle
import hashlib
import copy
import random

from verl.tools.mcp_base_tool import MCPBaseTool
from verl.tools.utils.mcp_clients.McpClientManager import ClientManager
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Suppress third-party library logging to prevent stack traces from being printed
# logging.getLogger('mcp').setLevel(logging.CRITICAL)
# logging.getLogger('httpx').setLevel(logging.CRITICAL) 
# logging.getLogger('httpcore').setLevel(logging.CRITICAL)
# logging.getLogger('fastmcp').setLevel(logging.CRITICAL)


class MCPChecklistTool(MCPBaseTool):
    # 类级别的数据缓存，键为dataset_path，值为解析后的数据
    _dataset_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()
    # 共享的 HTTP 客户端与并发控制
    _shared_client: httpx.AsyncClient | None = None
    _shared_semaphore: asyncio.Semaphore | None = None
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.return_raw: bool = bool(config.get("return_raw", True))

        self._dataset_path = config.get("dataset_path", None)
        
        # 从缓存或新建数据
        cached_data = self._get_or_load_dataset_data(self._dataset_path)
        self._id_by_tool_call_response = cached_data["id_by_tool_call_response"]
        self._id_by_candidate_tools = cached_data["id_by_candidate_tools"]
        self._id_by_candidate_tools_name = cached_data["id_by_candidate_tools_name"]
        self._tool_by_name = cached_data["tool_by_name"]
        self._tools = cached_data["tools"]

        self._sglang_url = config.get("sglang_url", [])
        self._sglang_model = config.get("sglang_model", None)
        # self._system_instruction = config.get("system_instruction", None) or (
        #     "You are a precise tool executor that learns from examples.\n"
        #     "You will be given:\n"
        #     "- Tool call JSON Schema\n"
        #     "- Few-shot examples showing tool calls and their execution results\n"
        #     "- A new tool call with specific arguments\n"
        #     "\n"
        #     "Your task:\n"
        #     "1) Learn the OUTPUT FORMAT from the provided examples - follow the exact structure, data types, and response patterns\n"
        #     "2) Ensure FACTUAL CONSISTENCY - your output should align with the factual information demonstrated in the examples\n"
        #     "3) For the new tool call:\n"
        #     "   - Apply the learned format to the new arguments\n"
        #     "   - Maintain factual consistency with example patterns\n"
        #     "   - If arguments are similar to examples, adapt the example results appropriately\n"
        #     "   - If arguments are significantly different, generate new results following the learned format and factual patterns\n"
        #     "   - May need to fix some type or error in the examples\n"
        #     "4) Handle errors gracefully - if arguments are invalid or missing, return error messages in the same format as examples\n"
        #     "\n"
        #     "Critical constraints:\n"
        #     "- Act as a silent function executor - NO explanations, suggestions, or hints\n"
        #     "- NO guidance on how to fix errors or improve calls\n"
        #     "- NO references to examples or comparisons\n"
        #     "- Return ONLY the raw execution result as valid JSON\n"
        #     "- For errors, return minimal error information without instructional content\n"
        #     "\n"
        #     "Output requirements:\n"
        #     "- Return ONLY the execution result as valid JSON\n"
        #     "- No explanations, markdown, or code fences\n"
        #     "- Use correct JSON data types\n"
        #     "- Follow the exact output structure learned from examples\n"
        #     "- Maintain factual consistency with the example patterns\n"
        # )
        self._system_instruction = config.get("system_instruction", None) or (
            "You are a precise tool executor that learns from examples.\n"
            "You will be given:\n"
            "- Tool call JSON Schema\n"
            "- Few-shot examples showing tool calls and their execution results\n"
            "- A new tool call with specific arguments\n"
            "\n"
            "Your task:\n"
            "1) Learn the OUTPUT FORMAT from the provided examples - follow the exact structure, data types, and response patterns\n"
            "2) Ensure FACTUAL CONSISTENCY - your output should align with the factual information demonstrated in the examples\n"
            "3) For the new tool call:\n"
            "   - Apply the learned format to the new arguments\n"
            "   - Maintain factual consistency with example patterns\n"
            "   - If arguments are similar to examples, adapt the example results appropriately\n"
            "   - If arguments are significantly different, generate new results following the learned format and factual patterns\n"
            "   - May need to fix some type or error in the examples\n"
            "4) Handle errors gracefully - if arguments are invalid or missing, return error messages in the same format as examples\n"
            "\n"
            "Critical constraints:\n"
            "- Act as a silent function executor - NO explanations, suggestions, or hints\n"
            "- NO guidance on how to fix errors or improve calls\n"
            "- NO references to examples or comparisons\n"
            "- Return ONLY the raw execution result as valid JSON\n"
            "- For errors, return minimal error information without instructional content\n"
            "\n"
            "Output requirements:\n"
            "- First do some analysis on how to mock the execution results. Then return ONLY the execution result as valid JSON array or object\n"
            "- No explanations, markdown, or code fences\n"
            "- Follow the exact output structure learned from examples\n"
            "- Maintain factual consistency with the example patterns\n"
            "Format:\n"
            "{\n"
            "  \"analysis\": str,\n"
            "  \"execution_result\": JSON array or object,\n"
            "}"
            ""
        )

        self._temperature = config.get("temperature", 0.6)
        self._max_new_tokens = config.get("max_new_tokens", 2048)
        self._json_retry_attempts = config.get("retry_attempts", 1)
        self._top_p = config.get("top_p", 0.8)
        self._max_tokens = config.get("max_tokens", 2048)
        self._timeout = config.get("timeout", 120)
        try:
            self._semaphore_size = int(config.get("semaphore_size", 64))
        except Exception:
            self._semaphore_size = 64

        # 初始化全局共享 client 和 semaphore（按首次实例的配置创建）
        try:
            timeout_value = float(self._timeout)
        except Exception:
            timeout_value = 120.0
        limits = httpx.Limits(
            max_connections=max(16, self._semaphore_size),
            max_keepalive_connections=max(8, self._semaphore_size // 2),
        )
        if MCPChecklistTool._shared_client is None:
            MCPChecklistTool._shared_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=timeout_value, read=timeout_value, write=timeout_value, connect=timeout_value),
                limits=limits,
            )
        if MCPChecklistTool._shared_semaphore is None:
            MCPChecklistTool._shared_semaphore = asyncio.Semaphore(self._semaphore_size)
    @classmethod
    def _get_or_load_dataset_data(cls, dataset_path: str) -> Dict[str, Any]:
        """获取或加载数据集数据，使用类级别缓存避免重复加载"""
        if dataset_path is None:
            raise ValueError("dataset_path cannot be None")
            
        # 使用线程锁确保线程安全
        with cls._cache_lock:
            if dataset_path in cls._dataset_cache:
                logger.info(f"Using cached dataset data for path: {dataset_path}")
                return cls._dataset_cache[dataset_path]
            
            logger.info(f"Loading and caching dataset data for path: {dataset_path}")
            
            # 可选：尝试从磁盘缓存加载（适合大数据集）
            disk_cache_data = cls._try_load_disk_cache(dataset_path)
            if disk_cache_data:
                logger.info(f"Loaded dataset from disk cache: {dataset_path}")
                cls._dataset_cache[dataset_path] = disk_cache_data
                return disk_cache_data
            
            # 加载数据
            data = cls._load_dataset_data(dataset_path)
            cls._dataset_cache[dataset_path] = data
            
            # 可选：保存到磁盘缓存
            cls._try_save_disk_cache(dataset_path, data)
                
            return data
    
    @staticmethod 
    def _load_dataset_data(dataset_path: str) -> Dict[str, Any]:
        """加载数据集并构建所需的数据结构"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        id_by_tool_call_response = {}
        id_by_candidate_tools = {}
        id_by_candidate_tools_name = {}
        tool_by_name = {}
        tools = []
        all_tools = []

        # 构建tool_call_response映射
        for item in data:
            tools = json.loads(item["extra_info"]["tools"])
            all_tools.extend(tools)
            tool_call_response_map = MCPChecklistTool._get_called_tools_and_response_static(item)
            id_by_tool_call_response[str(item["extra_info"]["original_index"])] = tool_call_response_map
        
        # 构建candidate_tools映射
        for item in data:
            tools = json.loads(item["extra_info"]["tools"])
            id_by_candidate_tools[str(item["extra_info"]["original_index"])] = {tool["function"]["name"]: tool for tool in tools}
            id_by_candidate_tools_name[str(item["extra_info"]["original_index"])] = [x["function"]["name"] for x in tools]

        # 构建tool映射
        all_tools_name = set([x["function"]["name"] for x in all_tools])
        for name in all_tools_name:
            mcp_tool = MCPTool(name=name, description="", inputSchema={"type": "object", "properties": {}, "required": []})
            tools.append(mcp_tool)
            tool_by_name[name] = mcp_tool

        return {
            "id_by_tool_call_response": id_by_tool_call_response,
            "id_by_candidate_tools": id_by_candidate_tools,
            "id_by_candidate_tools_name": id_by_candidate_tools_name,
            "tool_by_name": tool_by_name,
            "tools": tools
        }

    @staticmethod
    def _get_called_tools_and_response_static(item: Dict[str, Any]) -> List[Any]:
        """静态版本的get_called_tools_and_response方法，供数据加载使用"""
        if "extra_info" in item and "messages" in item["extra_info"]:
            messages = item["extra_info"]["messages"]
        else:
            raise ValueError(f"No messages found in item: {item}")
        
        results = defaultdict(list)
        if not isinstance(messages, list):
            raise ValueError(f"Unexpected messages format: {type(messages)}")

        for i in range(len(messages)):
            message = messages[i]
            if message.get("role") != "assistant":
                continue

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                continue

            # Collect contiguous following tool messages for this assistant turn
            following_tool_msgs: List[Dict] = []
            
            for j in range(i+1, i+len(tool_calls)+1):
                assert messages[j].get("role") == "tool" or messages[j].get("role") == "observation", f"Unexpected role: {messages[j].get('role')}"
                following_tool_msgs.append(messages[j]["content"])

            for call, content in zip(tool_calls, following_tool_msgs, strict=True):
                results[call["function"]["name"]].append((json.loads(call["function"]["arguments"]), content))
        return results

    def get_called_tools_and_response(self, item: Dict[str, Any]) -> List[Any]:
        """实例方法版本，调用静态方法实现"""
        return self._get_called_tools_and_response_static(item)
    
    @staticmethod
    def _sanitize_text_for_tokenizer(text: Any) -> Any:
        """Remove invalid Unicode surrogate code points that may break fast tokenizers or downstream consumers.

        This keeps valid non-ASCII characters intact while stripping only the surrogate range U+D800..U+DFFF.
        Accepts non-str inputs and returns them unchanged for convenience.
        """
        if not isinstance(text, str):
            return text
        has_surrogate = False
        sanitized_chars = []
        for ch in text:
            code = ord(ch)
            if 0xD800 <= code <= 0xDFFF:
                has_surrogate = True
                continue
            sanitized_chars.append(ch)
        if has_surrogate:
            try:
                logger.debug("[MCPChecklistTool] Stripped invalid surrogate code points from text.")
            except Exception:
                pass
        return "".join(sanitized_chars)
    
    def _format_error(self, code: str, message: str, details: Dict[str, Any] | None = None) -> str:
        """Return a standardized JSON error string."""
        payload: Dict[str, Any] = {
            "error_tool_call": {
                "code": code,
                "message": message,
            }
        }
        if details is not None:
            payload["error_tool_call"]["details"] = details
        # Ensure ASCII for safety with downstream JSON-only consumers
        return json.dumps(payload, ensure_ascii=True)

    def _validate_parameters_against_schema(self, original_index: str, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Lightweight validation of parameters against OpenAI-style tool schema from dataset.

        Checks:
        - required fields present
        - basic type conformity for primitive types
        - optional strict mode to disallow additional properties
        """
        errors: List[str] = []
        tool_schema: Dict[str, Any] = self._id_by_candidate_tools[original_index][self.name]

        fn = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else {}
        params_schema = fn.get("parameters", {}) if isinstance(fn, dict) else {}

        if not isinstance(parameters, dict):
            return False, ["parameters must be a JSON object"]

        properties: Dict[str, Any] = params_schema.get("properties", {}) if isinstance(params_schema, dict) else {}
        required: List[str] = params_schema.get("required", []) if isinstance(params_schema, dict) else []
        strict: bool = bool(fn.get("strict", True))

        # required fields
        for key in required:
            if key not in parameters:
                errors.append(f"missing required field: {key}")

        # type checks (primitive only)
        def _matches_type(value: Any, expected: Any) -> bool:
            if isinstance(expected, list):
                return any(_matches_type(value, t) for t in expected)
            if expected == "string":
                return isinstance(value, str)
            if expected == "number":
                return (isinstance(value, (int, float)) and not isinstance(value, bool))
            if expected == "integer":
                return (isinstance(value, int) and not isinstance(value, bool))
            if expected == "boolean":
                return isinstance(value, bool)
            if expected == "null":
                return value is None
            if expected == "object":
                return isinstance(value, dict)
            if expected == "array":
                return isinstance(value, list)
            # unknown type keywords are treated as pass
            return True

        for key, value in parameters.items():
            if key not in properties:
                if strict:
                    errors.append(f"unexpected field not allowed: {key}")
                continue
            prop = properties.get(key, {})
            expected_type = prop.get("type")
            if expected_type is not None and not _matches_type(value, expected_type):
                errors.append(f"field '{key}' type mismatch: expected {expected_type}")
            # enum constraint
            if "enum" in prop:
                enum_values = prop.get("enum")
                try:
                    # allow int/float equivalence only if exactly equal (no bool)
                    if isinstance(value, bool):
                        in_enum = value in enum_values
                    else:
                        in_enum = value in enum_values
                except Exception:
                    in_enum = False
                if not in_enum:
                    errors.append(f"field '{key}' not in enum: {enum_values}")

        return len(errors) == 0, errors
    
    @classmethod
    def clear_cache(cls, dataset_path: str = None):
        """清理缓存数据"""
        with cls._cache_lock:
            if dataset_path is None:
                # 清理所有缓存
                cls._dataset_cache.clear()
                logger.info("Cleared all dataset cache")
            else:
                # 清理特定路径的缓存
                if dataset_path in cls._dataset_cache:
                    del cls._dataset_cache[dataset_path]
                    logger.info(f"Cleared cache for dataset: {dataset_path}")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, int]:
        """获取缓存信息"""
        with cls._cache_lock:
            return {
                "cached_datasets": len(cls._dataset_cache),
                "dataset_paths": list(cls._dataset_cache.keys())
            }
    
    @staticmethod
    def _get_disk_cache_path(dataset_path: str) -> str:
        """生成磁盘缓存文件路径"""
        cache_dir = os.getenv("MCP_CACHE_DIR", "/tmp/mcp_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        path_hash = hashlib.md5(dataset_path.encode()).hexdigest()
        return os.path.join(cache_dir, f"dataset_{path_hash}.pkl")
    
    @staticmethod
    def _try_load_disk_cache(dataset_path: str) -> Dict[str, Any]:
        """尝试从磁盘缓存加载数据"""
        try:
            cache_path = MCPChecklistTool._get_disk_cache_path(dataset_path)
            if not os.path.exists(cache_path):
                return None
            
            # 检查缓存是否过期
            cache_mtime = os.path.getmtime(cache_path)
            dataset_mtime = os.path.getmtime(dataset_path)
            if cache_mtime < dataset_mtime:
                logger.info(f"Disk cache expired for {dataset_path}")
                return None
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load disk cache for {dataset_path}: {e}")
            return None
    
    @staticmethod  
    def _try_save_disk_cache(dataset_path: str, data: Dict[str, Any]):
        """尝试保存数据到磁盘缓存"""
        try:
            cache_path = MCPChecklistTool._get_disk_cache_path(dataset_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved dataset to disk cache: {dataset_path}")
        except Exception as e:
            logger.warning(f"Failed to save disk cache for {dataset_path}: {e}")
    def get_candidate_tools(self, item: Dict[str, Any]) -> List[Any]:
        messages = item["extra_info"]["messages"]
        
        
        results = defaultdict(list)
        if not isinstance(messages, list):
            raise ValueError(f"Unexpected messages format: {type(messages)}")

        for i in range(len(messages)):
            message = messages[i]
            if message.get("role") != "assistant":
                continue

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                continue

            # Collect contiguous following tool messages for this assistant turn
            following_tool_msgs: List[Dict] = []
            
            for j in range(i+1, i+len(tool_calls)+1):
                assert messages[j].get("role") == "tool" or messages[j].get("role") == "observation", f"Unexpected role: {messages[j].get('role')}"
                following_tool_msgs.append(messages[j]["content"])

            for call, content in zip(tool_calls, following_tool_msgs, strict=True):
                results[call["function"]["name"]].append((json.loads(call["function"]["arguments"]), content))
        return results

    async def execute(self, instance_id: str, parameters: dict[str, Any], original_index: str, **kwargs) -> tuple[ToolResponse, float, dict]:
        original_index = str(original_index)
        if original_index not in self._id_by_candidate_tools_name:
            msg = self._format_error(
                "INDEX_NOT_FOUND",
                f"original_index {original_index} not found",
                {"original_index": original_index},
            )
            logger.warning(f"[MCPTool] {msg}")
            return ToolResponse(text=msg), 0.0, {"success": False}
        if self.name not in self._id_by_candidate_tools_name[original_index]:
            msg = self._format_error(
                "TOOL_NOT_AVAILABLE",
                f"tool {self.name} not available for original_index {original_index}",
                {"tool": self.name, "original_index": original_index},
            )
            logger.warning(f"[MCPTool] {msg}")
            return ToolResponse(text=msg), 0.0, {"success": False}
        if original_index not in self._id_by_tool_call_response:
            msg = self._format_error(
                "INDEX_NOT_FOUND",
                f"original_index {original_index} not found in call responses",
                {"original_index": original_index},
            )
            logger.warning(f"[MCPTool] {msg}")
            return ToolResponse(text=msg), 0.0, {"success": False}

        if not self.name or parameters is None or not isinstance(parameters, dict):
            msg = self._format_error(
                "INVALID_PARAMETERS",
                "'parameters' is missing, empty, or not a JSON object.",
                {"tool": self.name, "parameters_type": type(parameters).__name__},
            )
            logger.warning(f"[MCPTool] {msg}")
            return ToolResponse(text=msg), 0.0, {"success": False}


        tool_call_response_map = self._id_by_tool_call_response[original_index]
        def _canonicalize_parameters(parameters: Dict[str, Any]) -> Any:
            # Keep ensure_ascii=False to preserve matching semantics with dataset; do not sanitize here
            return json.dumps(parameters, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        if self.name in tool_call_response_map:
            tool_call_response = tool_call_response_map[self.name]
            # find if arguments can match any of the tool call responses
            for args, content in tool_call_response:
                if _canonicalize_parameters(args) == _canonicalize_parameters(parameters):
                    logger.info(f"Found match for tool {self.name} with arguments {parameters}")
                    safe_text = (
                        self._sanitize_text_for_tokenizer(content)
                        if isinstance(content, str)
                        else json.dumps(content, ensure_ascii=True)
                    )
                    return ToolResponse(text=safe_text), 0.0, {"success": True}

        # Validate against schema before any attempt
        ok, validation_errors = self._validate_parameters_against_schema(original_index, parameters)
        if not ok:
            msg = self._format_error(
                "SCHEMA_VALIDATION_FAILED",
                "parameters do not conform to the tool schema",
                {"errors": validation_errors, "tool": self.name, "original_index": original_index},
            )
            logger.info(f"[MCPTool] Schema validation failed: {validation_errors}")
            return ToolResponse(text=msg), 0.0, {"success": True}


        # logger.info(f"No match for tool {self.name} with arguments {parameters}, but schema is correct, will call llm to mock the response.")
        # if valid, call llm to mock the response.
        # We have valid the tool name is unique, so we can get the schema from the candidate tools.
        schema_str = json.dumps(self._id_by_candidate_tools[original_index][self.name], ensure_ascii=True, indent=0)
        schema_str = self._sanitize_text_for_tokenizer(schema_str)

        # Build few-shot examples from this original index across all tool calls
        examples_lines = ["Previous tool calls and results (few-shot):"]
        try:
            for ex_tool_name, pairs in tool_call_response_map.items():
                for ex_args, ex_content in pairs:
                    # try:
                    ex_args_str = json.dumps(ex_args, ensure_ascii=True, indent=0)
                    ex_args_str = self._sanitize_text_for_tokenizer(ex_args_str)
                    # except Exception:
                    #     ex_args_str = str(ex_args)
                    if isinstance(ex_content, (dict, list)):
                        try:
                            ex_content_str = json.dumps(ex_content, ensure_ascii=True, indent=0)
                            ex_content_str = self._sanitize_text_for_tokenizer(ex_content_str)
                        except Exception:
                            raise Exception(f"Error: {ex_content}")
                    else:
                        assert isinstance(ex_content, str), f"Unexpected ex_content type: {type(ex_content)}"
                        ex_content_str = self._sanitize_text_for_tokenizer(ex_content)
                    examples_lines.append(
                        "Tool name:\n" + ex_tool_name + "\n"
                        + "Tool arguments:\n" + ex_args_str + "\n"
                        + "Tool execution result:\n" + ex_content_str +"\n"
                    )
        except Exception:
            logger.warning("Failed to build few-shot examples, using <unavailable> instead")
            examples_lines = ["Previous tool calls and results (few-shot): <unavailable>"]

        user_prompt = (
            "\n".join(examples_lines)
            + "\n\n"
            + "Current tool name: "
            + self.name
            + "\n"
            + "Current tool input schema (JSON Schema):\n"
            + schema_str
            + "\n"
            + "Current arguments (JSON):\n"
            + self._sanitize_text_for_tokenizer(json.dumps(parameters, ensure_ascii=True, indent=0))
            + "\n"
            + "Generate tool execution result in JSON format."
        )
        user_prompt = self._sanitize_text_for_tokenizer(user_prompt)

        base_messages = [
            {"role": "system", "content": self._system_instruction},
            {"role": "user", "content": user_prompt},
        ]

        async def _single_attempt(attempt_idx: int) -> tuple[int, str]:
            payload = {
                "model": self._sglang_model,
                "messages": base_messages,
                "temperature": self._temperature,
                "max_new_tokens": self._max_new_tokens,
                "max_tokens": self._max_tokens,
                "top_p": self._top_p,
                # "json_schema": {
                #     "type": ["object", "array"]   # 允许对象或数组
                # },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tool_execution_response",
                        "schema": {
                            "type": "object",
                            "required": ["analysis", "execution_result"],
                            "additionalProperties": False,
                            "properties": {
                                "analysis": {
                                    "type": "string",
                                },
                                "execution_result": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "minProperties": 1,
                                            "additionalProperties": True
                                        },
                                        {
                                            "type": "array",
                                            "items": {}
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },

                "sampling_params": {
                    "temperature": self._temperature,
                    "max_new_tokens": self._max_new_tokens,
                    "top_p": self._top_p,
                    "max_tokens": self._max_tokens,
                },


            }
            try:
                # 确保共享资源已初始化（惰性补充 + 健康检查）
                try:
                    timeout_value = float(self._timeout)
                except Exception:
                    timeout_value = 120.0
                limits = httpx.Limits(
                    max_connections=max(16, self._semaphore_size),
                    max_keepalive_connections=max(8, self._semaphore_size // 2),
                )
                client = MCPChecklistTool._shared_client
                # 如果 client 缺失或已关闭，则重建
                if client is None or getattr(client, "is_closed", False):
                    if client is not None:
                        try:
                            await client.aclose()
                        except Exception:
                            pass
                    MCPChecklistTool._shared_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(timeout=timeout_value, read=timeout_value, write=timeout_value, connect=timeout_value),
                        limits=limits,
                    )
                    client = MCPChecklistTool._shared_client
                if MCPChecklistTool._shared_semaphore is None:
                    MCPChecklistTool._shared_semaphore = asyncio.Semaphore(self._semaphore_size)

                selected_url = random.choice(self._sglang_url) if isinstance(self._sglang_url, list) and self._sglang_url else self._sglang_url
                async with MCPChecklistTool._shared_semaphore:
                    try:
                        resp = await client.post(selected_url, json=payload)
                    except (httpx.ReadError, httpx.HTTPError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, RuntimeError):  # type: ignore[attr-defined]
                        # 出现网络/关闭异常，重建 client 并重试一次
                        await asyncio.sleep(5)
                        try:
                            await client.aclose()
                        except Exception:
                            pass
                        MCPChecklistTool._shared_client = httpx.AsyncClient(
                            timeout=httpx.Timeout(timeout=timeout_value, read=timeout_value, write=timeout_value, connect=timeout_value),
                            limits=limits,
                        )
                        client = MCPChecklistTool._shared_client
                        resp = await client.post(selected_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                text = MCPChecklistTool._sanitize_text_for_tokenizer(text)
                finish_reason = choice.get("finish_reason", "unknown")
                return attempt_idx, (text, finish_reason)
            except Exception as e:  # type: ignore[attr-defined]
                # Return a JSON error object so upstream never crashes expecting JSON
                return attempt_idx, None

        # 串行重试方式 - 一个成功就立即返回
        fallback_text = None
        fallback_finish_reason = "all failed"
        error_message = None
        try:
            for attempt_idx in range(self._json_retry_attempts):
                try:
                    _, result = await _single_attempt(attempt_idx)
                    if result is None:
                        continue
                    text, finish_reason = result
                    # 保存第一个有效结果作为fallback
                    # logger.warning(finish_reason)
                    # if fallback_text is None and finish_reason == "stop":
                    #     fallback_text = text
                    #     fallback_finish_reason = "stop but failed to parse json"

                    parsed = json.loads(text)
                    tool_response = parsed['execution_result']
                    normalized = json.dumps(tool_response, ensure_ascii=True)
                    return ToolResponse(text=normalized), 0.0, {"success": True}
                except Exception as e:
                    error_message = str(e)
                    # logger.warning(f"Attempt {attempt_idx} failed with error: {e}, continuing...")
                    continue
        except Exception as e:
            logger.warning(f"Error during serial retry execution: {e}")
        
        # 所有尝试都失败了，返回错误信息
        if not fallback_text:
            fallback_text = f'{{"error": {error_message}}}'
            
        logger.warning(f"Tool {self.name} execution failed after {self._json_retry_attempts} serial attempts: {error_message}")
        return ToolResponse(text=fallback_text), 0.0, {"success": False}
        

       

        




        