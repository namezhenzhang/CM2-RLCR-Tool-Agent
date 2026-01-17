from typing import List, Dict, Any, Tuple
import json


def validate_tool_response_json_valid(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate whether each tool response content can be JSON-parsed
    
    Rules:
    - Each tool message content must be valid JSON format
    
    Args:
        messages: List of messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    for i, message in enumerate(messages):
        if message.get("role") == "tool":
            content = message.get("content", "")
            
            # Empty content is considered invalid
            if not content:
                return False, f"Tool message at index {i} has empty content"
            
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return False, f"Tool message at index {i} has invalid JSON content: {e}\nContent:\n{str(content)}"
            except Exception as e:
                return False, f"Tool message at index {i} failed to parse as JSON: {e}\nContent:\n{str(content)}"
    
    return True, ""
