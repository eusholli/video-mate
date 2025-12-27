import os
import dashscope
from http import HTTPStatus

# FIX 1: Add the International Endpoint URL (Just like in hello_qwen.py)
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# FIX 2: Remove the trailing comma at the end
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def test_key():
    # Print the type to prove it's a string now, not a tuple
    print(f"🔑 Key Type: {type(dashscope.api_key)}")
    print(f"🔑 Testing Alibaba Key: {str(dashscope.api_key)[:5]}...")
    
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        
        if response.status_code == HTTPStatus.OK:
            print("✅ SUCCESS! The key is working.")
            print(f"Response: {response.output.text}")
        else:
            print(f"❌ FAILED. Status Code: {response.status_code}")
            print(f"Message: {response.message}")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")

if __name__ == "__main__":
    test_key()
