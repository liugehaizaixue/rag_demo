from request_ollama import ORequest

PROMPT_TEMPLATE = """
你是一个智能问答助手，请基于以下已知信息与对话上下文，简洁和专业地来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

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
    messages = {"model": "qwen", "prompt": _content}
    
    ans = ORequest(messages)
    return ans


    
