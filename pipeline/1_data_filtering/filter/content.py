from typing import List, Dict, Any, Tuple


def validate_no_tool_response_in_assistant_message(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate that assistant messages should not contain tool response content
    
    Rules:
    - Assistant message content should not contain <tool_response> and </tool_response> tags
    
    Args:
        messages: List of messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    for i, message in enumerate(messages):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if not content:
                continue
            
            # Check if it contains tool_response tags
            if "<tool_response>" in content:
                return False, f"Assistant message at index {i} contains '<tool_response>' tag in content"
            
            if "</tool_response>" in content:
                return False, f"Assistant message at index {i} contains '</tool_response>' tag in content"
    
    return True, ""


def validate_non_empty_content(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate that any role has non-empty content
    
    Rules:
    - All message should have non-empty content
    
    Args:
        messages: List of messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    for i, message in enumerate(messages):
        content = message.get("content", "")
        if not content.strip():  # Check if content is empty after stripping whitespace
            return False, f"Message at index {i} has empty content"
    
    return True, ""


def validate_think_tags(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate think tags usage in messages:
    - User and tool messages should not contain <think></think> tags
    - Assistant messages should contain exactly one <think> and one </think> tag
    - <think> should come before </think> in assistant messages
    
    Args:
        messages: List of messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    for i, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role in ["user", "tool"]:
            # User and tool messages should not contain think tags
            if "<think>" in content or "</think>" in content:
                return False, f"{role.capitalize()} message at index {i} contains <think> or </think> tags"
        
        elif role == "assistant":
            # Assistant messages should have exactly one <think> and one </think>
            think_open_count = content.count("<think>")
            think_close_count = content.count("</think>")
            
            if think_open_count != 1:
                return False, f"Assistant message at index {i} has {think_open_count} <think> tags, expected exactly 1"
            
            if think_close_count != 1:
                return False, f"Assistant message at index {i} has {think_close_count} </think> tags, expected exactly 1"
            
            # Check that <think> comes before </think>
            think_open_pos = content.find("<think>")
            think_close_pos = content.find("</think>")
            
            if think_open_pos >= think_close_pos:
                return False, f"Assistant message at index {i} has <think> at position {think_open_pos} which is not before </think> at position {think_close_pos}"
    
    return True, ""
