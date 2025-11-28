import re
from typing import List, Tuple, Optional, Dict, Any
import json

def parse_llm_output(output: str) -> Tuple[Optional[str], Optional[str], Optional[str | Dict[str, Any]]]:
    regex = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    match : List[str] = regex.findall(output)
    if len(match) > 1:
        raise MultipleToolCallException()
    if len(match) == 0:
        return None, None, None
    tool_call = match[0]
    try:
        tool_call_json = json.loads(tool_call)
        return tool_call_json["context"], tool_call_json["tool"], tool_call_json.get("args", {})
    except Exception as e:
        raise InvalidToolCallJSONException()
    

if __name__ == "__main__":
    output = """🤔 I'm going to open the "TEXT-default-hint" window to read the hint again and then use the end tool to end the task.
<tool_call>{'context': 'TEXT-default-hint', 'tool': 'read'}</tool_call>"""
    print(parse_llm_output(output))