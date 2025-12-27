import os
import dashscope
from dashscope import MultiModalConversation
from dashscope.audio.asr import Recognition
import wave
import struct
from http import HTTPStatus

# ====================================================================
# 1. CRITICAL CONFIGURATION FIXES
# ====================================================================
# Force the SDK to use the International (Singapore) Endpoints.
# If these are missing, the SDK defaults to China (Beijing) and fails with 401.

# HTTP Endpoint (Used by Qwen / Chat Models)
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# WebSocket Endpoint (Used by Paraformer / Audio Models)
# Note: The SDK defaults this to 'wss://dashscope.aliyuncs.com...', which rejects international keys.
dashscope.base_websocket_api_url = 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/'

# Load API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("⚠️ WARNING: DASHSCOPE_API_KEY not found in environment.")
    # You can uncomment this line for quick testing if env vars fail:
    # api_key = "sk-YOUR_ACTUAL_KEY_HERE"

dashscope.api_key = api_key

# ====================================================================
# 2. HELPER: Create Dummy Audio
# ====================================================================
def create_silent_wav(filename="test_silence.wav", duration=1.0):
    """Generates a 1-second silent WAV file for testing ASR."""
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        # Write silence (zeros)
        wav_file.writeframes(struct.pack('<' + ('h' * num_samples), *[0] * num_samples))
    
    return filename

# ====================================================================
# 3. TEST FUNCTIONS
# ====================================================================
def test_caption_model():
    print(f"\n📸 TESTING CAPTION MODEL: qwen-vl-plus-latest")
    print("-" * 50)
    
    try:
        # Simple text-only query to check permission
        messages = [{
            'role': 'user',
            'content': [
                {'text': 'Describe a cat in one sentence.'}
            ]
        }]
        
        response = MultiModalConversation.call(
            model='qwen-vl-plus-latest', # Or 'qwen-vl-plus'
            messages=messages
        )
        
        if response.status_code == HTTPStatus.OK:
            print("✅ SUCCESS! Access granted to Qwen-VL.")
            # Depending on SDK version, output structure varies slightly.
            # Using safe access:
            try:
                content = response.output.choices[0].message.content[0]['text']
                print(f"Output: {content}")
            except:
                print(f"Output (Raw): {response}")
        else:
            print(f"❌ FAILED. Status: {response.status_code}")
            print(f"Message: {response.message}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_asr_model():
    print(f"\n🎙️ TESTING ASR MODEL: 'fun-asr-realtime'")
    print("-" * 50)
    
    # Generate dummy file
    audio_file = create_silent_wav()
    
    try:
        recognition = Recognition(
            model='fun-asr-realtime',
            format='wav',
            sample_rate=16000,
            callback=None 
        )
        
        # The SDK will now use dashscope.base_websocket_api_url automatically
        response = recognition.call(audio_file)
        
        if response.status_code == HTTPStatus.OK:
            print("✅ SUCCESS! Access granted to Paraformer.")
            print(f"Server recognized audio: {response}")
        else:
            print(f"❌ FAILED. Status: {response.status_code}")
            print(f"Message: {response.message}")
            if response.status_code == 401:
                print("DEBUG: The WebSocket URL might still be pointing to China.")
                print(f"Current WS URL: {dashscope.base_websocket_api_url}")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        # Clean up
        if os.path.exists(audio_file):
            os.remove(audio_file)

# ====================================================================
# 4. MAIN EXECUTION
# ====================================================================
if __name__ == "__main__":
    print(f"🔑 Key Check: {str(api_key)[:5]}...")
    print(f"🌍 HTTP Endpoint: {dashscope.base_http_api_url}")
    print(f"🌍 WS Endpoint:   {dashscope.base_websocket_api_url}")
    
    test_caption_model()
    test_asr_model()
