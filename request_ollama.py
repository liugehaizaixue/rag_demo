import requests
import json

url = "http://localhost:11434/api/generate"
headers = {
    'Content-Type': 'application/json'
}

models = ['qwen','llama3.2']

# data = {
#     "model": "qwen",
#     "prompt": "Why is the sky blue?"
# }

def ORequest(data):
    # 发送 POST 请求并启用流模式
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    ans = ""
    # 检查响应状态码
    if response.status_code == 200:
        # 逐行读取响应
        for line in response.iter_lines(decode_unicode=True):
            if line:  # 确保行不为空
                try:
                    # 解析 JSON 对象
                    json_obj = json.loads(line)
                    ans += json_obj.get("response")
                except json.JSONDecodeError:
                    print("Error decoding JSON:", line)
    else:
        print("Request failed with status code:", response.status_code)

    print(ans)
    return ans

if __name__ == "__main__":
    ORequest(data)