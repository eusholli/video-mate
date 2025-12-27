import os
from dashscope import Generation
import dashscope
# The following URL is for the Singapore region. If you use a model in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Who are you?'}
    ]
response = Generation.call(
    # The API keys for the Singapore and China (Beijing) regions are different. To create an API key, see https://modelstudio.console.alibabacloud.com/?tab=model#/api-key
    # If you have not exported an environment variable, replace the following line with your Model Studio API key: api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    model="qwen-plus",   
    messages=messages,
    result_format="message"
)

if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    print(f"HTTP return code: {response.status_code}")
    print(f"Error code: {response.code}")
    print(f"Error message: {response.message}")
    print("For more information, see https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")
