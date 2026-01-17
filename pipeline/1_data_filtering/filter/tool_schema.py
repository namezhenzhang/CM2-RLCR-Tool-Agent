import json
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure logger to show line numbers with colors
handler = logging.StreamHandler()
formatter = logging.Formatter('\033[92m%(asctime)s\033[0m - \033[94m%(name)s\033[0m - \033[93m%(levelname)s\033[0m - \033[95m%(filename)s:%(lineno)d\033[0m - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class ToolSchemaValidator:
    """Tool schema validator for validating assistant's tool calls against provided tool schema"""
    
    def __init__(self, tools: List[Dict[str, Any]]):
        """
        Initialize validator
        
        Args:
            tools: List containing tools
        """
        self.available_tools = defaultdict(list)
        self.is_valid, self.error_message = self._parse_tools_config(tools)
    
    def _parse_tools_config(self, tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Parse tool configuration and extract schema of all available tools"""
        # tools = tools_config.get('tools', [])
        for tool in tools:
            if tool.get('type') == 'function' and 'function' in tool:
                func_info = tool['function']
                func_name = func_info.get('name')
                if func_name:
                    parameters = func_info.get('parameters', {})
                    # Check if it's an empty dictionary {}
                    if not parameters:
                        parameters = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    self.available_tools[func_name].append({
                        "description": func_info.get('description', ''),
                        "parameters": parameters
                    })
                else:
                    return False, f"Candidate tool name is not valid: {func_name}"
            else:
                return False, f"Candidate tool {tool} is not a valid tool: function not found"
        return True, ""
    
    def validate_tool_calls_from_messages(self, messages: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate whether assistant's tool calls in messages all conform to schema
        
        Args:
            messages: List of messages
            
        Returns:
            Tuple[bool, str]: If all tool calls conform to schema return (True, ""), otherwise return (False, error_message)
        """
        for message in messages:
            if message.get('role') == 'assistant':
                tool_calls = message.get('tool_calls', [])
                if tool_calls:
                    # If there are tool calls, need to validate each one
                    for tool_call in tool_calls:
                        if tool_call.get('type') != 'function':
                            return False, f"Tool call {tool_call} is not a valid tool call: type is not function"
                        try:
                            arguments = tool_call['function']['arguments']
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                            else:
                                arguments = arguments
                        except json.JSONDecodeError:
                            return False, f"Decode error for tool call arguments: {arguments}\nTool call: {tool_call}"
                        except Exception as e:
                            return False, f"Exception for tool call arguments: {e}\nTool call: {tool_call}"
                        formatted_tool_call = {
                            "name": tool_call['function']['name'],
                            "arguments": arguments
                        }
                        valid, error = self.validate_single_tool_call(formatted_tool_call)
                        if not valid:
                            return False, error
        return True, ""
    
    def validate_single_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate whether a single tool call conforms to schema"""


        function_name = tool_call.get('name')
        arguments = tool_call.get('arguments')

        # Check if function name is in available tools
        if function_name not in self.available_tools:
            return False, f"Tool call {tool_call} is not a valid tool call: Function name {function_name} is not in the available tools"

        # Get tool schema
        tool_schema_list = self.available_tools[function_name]
        error = ""
        # Validate if parameters conform to schema, one match is sufficient
        for tool_schema in tool_schema_list:
            valid, error = self.validate_parameters(arguments, tool_schema.get('parameters', {}))
            if valid:
                return True, ""
            else:
                continue
        return False, f"Tool call {tool_call} is not a valid tool call: No available tool schema found for function name {function_name}, error: {error}"
    
    def validate_parameters(self, arguments: Dict[str, Any], param_schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate whether parameters conform to schema"""
        if param_schema.get('type') != 'object':
            logger.warning(f"Parameter schema {param_schema} is not a valid object: type is not object")
            return False, f"Parameter schema {param_schema} is not a valid object: type is not object"  # If not object type, consider invalid for now
        
        properties = param_schema.get('properties', {})
        required_params = param_schema.get('required', [])
        
        # Check if all required parameters exist
        for required_param in required_params:
            if required_param not in arguments:
                return False, f"Required parameter {required_param} of {param_schema} is not found in the arguments {arguments}"
        
        # Check parameter types (simplified version, only check basic types)
        for param_name, param_value in arguments.items():
            if param_name in properties:
                expected_type = properties[param_name].get('type')
                if not self._check_param_type(param_value, expected_type):
                    return False, f"arguments: {arguments}, parameter {param_name} is not a valid {expected_type} in {param_schema}"
            else:
                return False, f"arguments: {arguments}, parameter {param_name} is not in the properties of {param_schema}"
        
        return True, ""
    
    def _check_param_type(self, value: Any, expected_type) -> Tuple[bool, str]:
        """Check if parameter type matches"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }

        try:
            # Handle case where expected_type is a list of types (e.g., ['string', 'number'])
            if isinstance(expected_type, list):
                for single_type in expected_type:
                    expected_python_type = type_mapping.get(single_type)
                    if expected_python_type and isinstance(value, expected_python_type):
                        return True, ""
                return False, f"Type of parameter {value} is not a valid {expected_type}"
            
            # Handle case where expected_type is a single type string
            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type is None:
                raise ValueError(f"Expected type {expected_type} is not a valid type")
            
            if isinstance(value, expected_python_type):
                return True, ""
            else:
                return False, f"Type of parameter {value} is not a valid {expected_type}"
        except Exception as e:
            raise ValueError(f"Expected type {expected_type} is not a valid type")
    
    def get_available_tools(self) -> List[str]:
        """Get list of all available tool names"""
        return list(self.available_tools.keys())


def validate_data_sample(data_sample: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate single data sample
    
    Args:
        data_sample: Data sample containing messages and tools
        
    Returns:
        bool: Validation result, True means conforms to schema, False means does not conform
    """
    tools = data_sample.get('tools', [])
    messages = data_sample.get('messages', [])

    validator = ToolSchemaValidator(tools)

    if not validator.is_valid:
        # logger.warning(f"Tools config is not valid: {validator.error_message}")
        return False, validator.error_message

    valid, error_message = validator.validate_tool_calls_from_messages(messages)
    if not valid:
        # logger.warning(f"Tool calls are not valid: {error_message}")
        return False, error_message
    return True, ""
