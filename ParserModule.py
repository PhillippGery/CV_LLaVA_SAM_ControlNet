# ParserModule.py
import json
from typing import Optional, Dict, Any

# Define the expected structure for a tool call
TOOL_NAMES = {"SAM", "DIFFUSION"}

def parse_tool_call(llm_output: str) -> Optional[Dict[str, Any]]:
    """
    Searches the LLM's output for a structured JSON tool call and validates it.

    Args:
        llm_output: The raw text output from the LMM.

    Returns:
        A dictionary containing the parsed tool call (name and arguments), 
        or None if no valid tool call is found.
    """
    # The LMM's output might be surrounded by text, so we look for the JSON block.
    # Simple heuristic: find the first '{' and the last '}'
    try:
        start = llm_output.find('{')
        end = llm_output.rfind('}')

        if start == -1 or end == -1:
            return None # No JSON found

        json_str = llm_output[start : end + 1]

        # Attempt to parse the JSON string
        tool_call_data = json.loads(json_str)

        # --- Validation Checks ---

        # 1. Check for required keys
        if 'tool_name' not in tool_call_data or 'arguments' not in tool_call_data:
            print("Parser Error: Missing 'tool_name' or 'arguments' in JSON.")
            return None

        # 2. Check for valid tool name
        tool_name = tool_call_data['tool_name'].upper()
        if tool_name not in TOOL_NAMES:
            print(f"Parser Error: Invalid tool_name '{tool_name}'. Must be one of {TOOL_NAMES}.")
            return None

        # 3. Check arguments type
        if not isinstance(tool_call_data['arguments'], dict):
            print("Parser Error: 'arguments' must be a JSON object (dictionary).")
            return None

        # Standardize the tool name (e.g., ensure it's uppercase)
        return {
            "tool_name": tool_name,
            "arguments": tool_call_data['arguments']
        }

    except json.JSONDecodeError:
        print("Parser Error: Could not decode the found JSON string.")
        return None
    except Exception as e:
        print(f"An unexpected parsing error occurred: {e}")
        return None