from openai import OpenAI 
from typing import Dict, List, Literal, Union
import json

def api_call_format(
             messages: List[Dict[str, str]], 
             api_url_config: dict,
             inference_config: dict,
             system_prompt: str = None
             ) -> str:
    '''
    - 使用 openai call api (统一接口，支持多轮)
    - messages: List[Dict], 对话历史列表
    - system_prompt: 可选系统提示词
    '''
    client = OpenAI(**api_url_config) 

    # 浅拷贝，避免修改外部传入的列表
    final_messages = list(messages)
    
    # 插入 System Prompt (如果列表开头不是 system)
    if system_prompt:
        if not final_messages or final_messages[0].get("role") != "system":
            final_messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        completion = client.chat.completions.create(
            messages=final_messages,
            **inference_config
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API Call Error: {e}")
        return "{}" # 返回空JSON结构防止程序崩溃