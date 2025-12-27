import os
from openai import OpenAI

# 1. Print proxy env vars to see if Python sees them
print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

try:
    # We use a tiny model just to test the connection
    print("Attempting connection...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}]
    )
    print("✅ Success! Response:", response.choices[0].message.content)
except Exception as e:
    print(f"❌ Failed: {type(e).__name__}")
    print(e)
