from typing import List, Dict, Any, Tuple


def validate_role_order(messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate whether the role order in messages follows the rules.
    
    Rules:
    1. If system appears, it must be the first one.
    2. If the first is system, the second must be user; otherwise, the first is user.
    3. user can only be followed by assistant.
    4. assistant can be followed by tool or user.
    5. tool can be followed by tool or assistant.
    6. The last message must be assistant.
    7. The last message must not have tool_calls.
    8. Conversation must contain at least one complete turn: user -> assistant.
    9. If an assistant has tool_calls, the next message must be tool.
    10. If an assistant has no tool_calls, it must not be followed by tool.
    
    Args:
        messages: List of messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not messages:
        return False, "Messages list is empty"
    
    roles = [msg.get("role") for msg in messages]
    
    # Rule 1: If system appears, it must be the first one
    for i, role in enumerate(roles):
        if role == "system" and i != 0:
            return False, f"System role found at position {i}, but it must be at position 0"
    
    # Rule 2: If the first is system, the second must be user; otherwise, the first is user
    if roles[0] == "system":
        if len(roles) < 2:
            return False, "If first message is system, there must be at least 2 messages"
        if roles[1] != "user":
            return False, f"If first message is system, second message must be user, but got {roles[1]}"
    elif roles[0] != "user":
        return False, f"First message must be user or system, but got {roles[0]}"
    
    # Rules 3-5: Check if the role after each message follows the rules
    for i in range(len(roles) - 1):
        current_role = roles[i]
        next_role = roles[i + 1]
        
        if current_role == "user":
            # Rule 3: user can only be followed by assistant
            if next_role != "assistant":
                return False, f"User at position {i} must be followed by assistant, but got {next_role}"
        
        elif current_role == "assistant":
            # Rule 4: assistant can be followed by tool or user
            if next_role not in ["tool", "user"]:
                return False, f"Assistant at position {i} can only be followed by tool or user, but got {next_role}"
            # Rule 9: If assistant has tool_calls, it must be followed by tool
            has_tool_calls = bool(messages[i].get("tool_calls"))
            if has_tool_calls and next_role != "tool":
                return False, (
                    f"Assistant at position {i} has tool_calls and must be followed by tool, but got {next_role}"
                )
            # Rule 10: If assistant does not have tool_calls, it must not be followed directly by tool
            if not has_tool_calls and next_role == "tool":
                return False, (
                    f"Assistant at position {i} has no tool_calls and must not be followed by tool"
                )
        
        elif current_role == "tool":
            # Rule 5: tool can be followed by tool or assistant
            if next_role not in ["tool", "assistant"]:
                return False, f"Tool at position {i} can only be followed by tool or assistant, but got {next_role}"
        
        elif current_role == "system":
            # system can only be in the first position, already checked in Rule 1
            continue
        
        else:
            return False, f"Unknown role {current_role} at position {i}"
    
    # Rule 6: The last message must be assistant
    if messages[-1].get("role") != "assistant":
        return False, f"Last message must be assistant, but got {messages[-1].get('role')}"
    
    # Rule 7: The last message must not have tool_calls
    if messages[-1].get("tool_calls"):
        return False, f"Last message must not have tool_calls, but got {messages[-1].get('tool_calls')}"

    # Rule 8: There must be at least one complete interaction user -> assistant
    has_complete_turn = any(
        roles[i] == "user" and roles[i + 1] == "assistant" for i in range(len(roles) - 1)
    )
    if not has_complete_turn:
        return False, "Conversation must contain at least one complete turn: user -> assistant"
    
    return True, ""
