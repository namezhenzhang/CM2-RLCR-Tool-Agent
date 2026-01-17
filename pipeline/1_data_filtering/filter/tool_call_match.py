from typing import List, Dict, Any, Tuple


def validate_tool_call_response_match(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate whether tool calls and tool responses match
    
    Rules:
    - For each assistant message with N tool_calls, there must be N tool messages immediately following
    - No need to match specific tool_call_id, only ensure count correspondence
    
    Args:
        messages: List of messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    i = 0
    while i < len(messages):
        message = messages[i]
        
        # If it's an assistant message with tool_calls
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                expected_tool_count = len(tool_calls)
                
                # Check the number of immediately following tool messages
                actual_tool_count = 0
                j = i + 1
                
                # Calculate the number of consecutive tool messages
                while j < len(messages) and messages[j].get("role") == "tool":
                    actual_tool_count += 1
                    j += 1
                
                # Verify if the counts match
                if actual_tool_count != expected_tool_count:
                    return False, f"Assistant at index {i} has {expected_tool_count} tool calls but followed by {actual_tool_count} tool responses"
                
                # Skip the already checked tool messages
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return True, ""
