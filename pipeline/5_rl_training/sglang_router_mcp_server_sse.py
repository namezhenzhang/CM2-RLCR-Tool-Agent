import argparse
import json
import os
from typing import Any, List

from mcp.server.fastmcp.server import FastMCP
from mcp.types import Tool as MCPTool
import logging

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# def load_tools(tools_json_path: str) -> List[MCPTool]:
#     with open(tools_json_path, "r", encoding="utf-8") as f:
#         raw = json.load(f)

#     tools: List[MCPTool] = []
#     for item in raw:
#         if not isinstance(item, dict):
#             continue
#         if item.get("type") != "function":
#             continue
#         fn = item.get("function", {})
#         name = fn.get("name")
#         description = fn.get("description", "")
#         parameters = fn.get("parameters", {"type": "object", "properties": {}, "required": []})
#         tool = MCPTool(name=name, description=description, inputSchema=parameters)
#         tools.append(tool)
#     return tools

# def get_all_tools(dataset_path: str) -> List[MCPTool]:
#     with open(dataset_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     all_tools = []
#     for item in data:
#         tools = tool_utils.get_tools(item)
#         all_tools.extend(tools)

#     all_tools = tool_utils.dedupe_and_validate_tools(all_tools)

class SGLangRouterMCPServer(FastMCP[Any]):
    def __init__(
        self,
        *,
        dataset_path: str,
        sglang_url: str,
        sglang_model: str,
        host: str = "0.0.0.0",
        port: int = 18080,
        system_instruction: str | None = None,
        temperature: float = 0.6,
        retry_attempts: int = 3,
        max_generated_tokens: int = 100,
    ) -> None:
        super().__init__(
            name="sglang-router-mcp",
            host=host,
            port=port,
        )
        self._dataset_path = dataset_path
        self._tools = []
        self.init_tools(dataset_path)

        self._sglang_url = sglang_url.rstrip("/")
        self._sglang_model = sglang_model
        self._system_instruction = system_instruction or (
            "You are a precise tool executor.\n"
            "You will be given: previous tool call examples (few-shot), a tool's JSON Schema, "
            "and the current tool name with JSON arguments.\n"
            "Your task:\n"
            "1) Return ONLY the final result for the current call as a single valid JSON object.\n"
            "2) Strictly follow the provided schema and arguments; infer reasonable values if needed.\n"
            "3) No explanations, no extra text, no markdown, no code fences.\n"
            "4) Use correct JSON types; avoid NaN/Infinity; no trailing commas.\n"
            "5) Do not include keys not described by the schema (unless clearly allowed).\n"
            "6) Ensure the output is factually consistent with the provided examples; do not contradict them."
            "7) Return ONLY a valid JSON object for the current call as tool result. No explanations, no markdown, no code fences."
        )
        self._temperature = temperature
        self._max_generated_tokens = max_generated_tokens
        self._json_retry_attempts = retry_attempts

    def init_tools(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            data = json.load(f)
        all_tools = []

        for item in data:
            tools = json.loads(item["extra_info"]["tools"])
            all_tools.extend(tools)
        # Handle both OpenAI format ({"type": "function", "function": {...}}) and simple format ({"name": ...})
        all_tools_name = set([
            x["function"]["name"] if "function" in x else x["name"] 
            for x in all_tools
        ])
        for name in all_tools_name:
            mcp_tool = MCPTool(name=name, description="", inputSchema={"type": "object", "properties": {}, "required": []})
            self._tools.append(mcp_tool)

    async def list_tools(self) -> List[MCPTool]:
        return self._tools


def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang Router MCP SSE Server")
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", 18080)))
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "DATASET_PATH",
            os.path.join(os.path.dirname(__file__), "data.json"),
        ),
        help="Path to dataset file",
    )
    parser.add_argument(
        "--sglang-url",
        default=os.environ.get("SGLANG_URL", "http://127.0.0.1:8000"),
        help="Base URL to SGLang server (without trailing slash)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SGLANG_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        help="Model name for SGLang",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for SGLang")
    parser.add_argument("--max-generated-tokens", type=int, default=2048, help="Max generated tokens for SGLang")
    parser.add_argument("--retry-attempts", type=int, default=1, help="Retry attempts for SGLang")
    args = parser.parse_args()

    server = SGLangRouterMCPServer(
        dataset_path=args.dataset_path,
        sglang_url=args.sglang_url,
        sglang_model=args.model,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        retry_attempts=args.retry_attempts,
        max_generated_tokens=args.max_generated_tokens,
    )

    # Run SSE transport (default endpoints: /sse and /messages/)
    print(f"Running SSE transport on port {args.port}")
    server.run("sse")




if __name__ == "__main__":
    main()


