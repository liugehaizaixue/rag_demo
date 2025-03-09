from request_ollama import ORequest
from request_qwen import get_response

PROMPT_TEMPLATE = """
你是一个智能问答助手，请基于以下已知信息与对话上下文，简洁和专业地来回答用户的问题。

已知信息:
{context_str}

对话历史:
{chat_history}

问题:
{query_str}
"""

def get_ai_answer(chat_history, context_str, query_str):
    chat = ""
    for message in chat_history:
        chat += f"{message['role']}: {message['content']}\n"

    _content = PROMPT_TEMPLATE.format(context_str=context_str, chat_history=chat, query_str=query_str)


    # 在线模型
    messages = [{
        "role": "system",
        "content":_content
    }]
    ans = get_response(messages).output.choices[0].message.content
    # ========================
    # 本地模型
    # messages = {"model": "qwen", "prompt": _content}
    # ans = ORequest(messages)

    return ans


    
