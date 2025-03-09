import os
from dashscope import Generation
import dashscope

def get_response(messages):
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-dea50ab4b6e34bc0bf9c937b19c11d08",
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-max",
        messages=messages,
        result_format="message",
    )
    return response

# # 初始化一个 messages 数组
# messages = [
#     {
#         "role": "system",
#         "content": """""",
#     }
# ]

# assistant_output = get_response(messages).output.choices[0].message.content

