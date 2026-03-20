import json
import json
import re
import random
import os
from datetime import datetime
from collections import Counter
from typing import List, Dict
import ast

def extract_json(response: str, key: str = None):
    """
    从文本中提取 JSON 或 Python 字典对象。
    (代码保持不变)
    """
    if not response:
        return None

    text = re.sub(r'^```\w*\n|```$', '', response.strip(), flags=re.MULTILINE).strip()
    candidates = []
    stack_count = 0
    start_index = -1
    in_string = False
    escape = False
    
    for i, char in enumerate(text):
        if char == '"' or char == "'":
            if not escape:
                in_string = not in_string
            escape = False
        elif char == '\\' and in_string:
            escape = not escape
        elif not in_string:
            if char == '{':
                if stack_count == 0:
                    start_index = i
                stack_count += 1
            elif char == '}':
                if stack_count > 0:
                    stack_count -= 1
                    if stack_count == 0:
                        candidates.append(text[start_index : i+1])
    
    def _process_result(result_obj):
        if key is not None:
            if isinstance(result_obj, dict):
                return result_obj.get(key)
            return None
        return result_obj

    if candidates:
        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
                return _process_result(obj)
            except json.JSONDecodeError:
                pass
            
            try:
                unescaped_candidate = candidate.replace(r'\"', '"')
                obj = json.loads(unescaped_candidate)
                return _process_result(obj)
            except json.JSONDecodeError:
                pass

            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, (dict, list)):
                    return _process_result(obj)
            except (ValueError, SyntaxError):
                pass

    try:
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        for match in reversed(matches):
            try:
                obj = json.loads(match)
                return _process_result(obj)
            except:
                pass
            try:
                obj = json.loads(match.replace(r'\"', '"'))
                return _process_result(obj)
            except:
                pass
            try:
                obj = json.loads(match.replace("'", '"'))
                return _process_result(obj)
            except:
                continue
    except:
        pass

    return None