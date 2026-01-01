import os
import asyncio
import json
from tqdm import tqdm
import dashscope
from dashscope.audio.asr import Recognition
from .._utils import logger
import subprocess
import uuid

async def process_single_segment(semaphore, index, segment_name, audio_file, model, audio_output_format, sample_rate):
    """
    Process a single audio segment with ASR
    """
    api_json_file = os.path.splitext(audio_file)[0] + ".json"
    
    # Check for cached transcript (JSON format first)
    if os.path.exists(api_json_file):
        try:
            with open(api_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Cache check modified to be lenient as we might fallback to non-diarized model
            if data and data.get('text'):
                logger.info(f"Using cached transcript (JSON) for {segment_name}")
                return index, data
        except Exception:
            pass

    # Legacy text file check (fallback, but we probably want to re-process to get timestamps if missing)
    # For now, if only text exists, we might miss timestamps. 
    # Let's assume we proceed to processing if JSON is missing to ensure we get timestamps.

    async with semaphore:  # Limit concurrent requests
        # Create temp file for conversion (Mono 16k)
        temp_audio = f"{audio_file}_{uuid.uuid4().hex}.mp3"
        
        result = None # Initialize result
        try:
            logger.info(f"Processing segment {segment_name} with model {model}")
            # Convert to Mono 16kHz using ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", audio_file,
                "-ac", "1", "-ar", "16000",
                "-loglevel", "error",
                temp_audio
            ]
            subprocess.run(cmd, check=True)
            
            # Use temp file for recognition
            recognition = Recognition(
                model=model,
                format="mp3", # We converted to mp3
                sample_rate=16000,
                language_hints=['en'],
                diarization_enabled=False, # Not supported on fun-asr-realtime
                callback=None  # type: ignore  # SDK type annotation issue
            )
            
            # Call the API
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, recognition.call, temp_audio)
            
        except Exception as e:
            logger.error(f"ASR failed for segment {segment_name}: {e}")
            result = None
        finally:
             if os.path.exists(temp_audio):
                 os.remove(temp_audio)

        if result and result.status_code == 200:
            # Extract text and sentences from result
            text = ""
            if "payload" in result.output:
                 # Realtime model structure
                 # Note: structure might differ, usually output has 'sentence'
                 pass
            
            # Standard Recognition output
            sentences_data = result.output.get("sentence", [])
            full_text = ""
            detailed_sentences = []
            
            for sentence in sentences_data:
                s_text = sentence.get('text', '')
                full_text += s_text + " "
                detailed_sentences.append({
                    "text": s_text,
                    "start": sentence.get('begin_time'), # ms
                    "end": sentence.get('end_time'),     # ms
                    "speaker_id": sentence.get('speaker_id') # Might be None
                })
            
            response_json = {
                "text": full_text.strip(),
                "sentences": detailed_sentences
            }
            
            # Cache the result as JSON
            try:
                with open(api_json_file, 'w', encoding='utf-8') as f:
                    json.dump(response_json, f, ensure_ascii=False, indent=2)
            except Exception as e:
                 logger.error(f"Failed to save JSON cache: {e}")
                 
            return index, response_json
        else:
            msg = result.message if result else "Unknown error"
            logger.error(f"ASR Task failed for segment {segment_name}: {msg}")
            return index, {"text": "", "sentences": []}

async def speech_to_text_online(video_name, working_dir, segment_index2name, audio_output_format, global_config, max_concurrent=5):
    """
    Online ASR using Alibaba Cloud DashScope API with async concurrent processing
    """
    # Get API key and sample rate from global config
    api_key = global_config.get('ali_dashscope_api_key')
    sample_rate = global_config.get('audio_sample_rate', 16000)
    
    # Set the API key
    dashscope.api_key = api_key
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all segments
    tasks = []
    for index in segment_index2name:
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        
        task = process_single_segment(
            semaphore, index, segment_name, audio_file, 
            global_config.get('asr_model'), audio_output_format, sample_rate
        )
        tasks.append(task)
    
    # Execute all tasks concurrently with real-time progress
    total_tasks = len(tasks)
    logger.info(f"🎤 Starting ASR for {total_tasks} audio segments (max {max_concurrent} concurrent)...")
    
    transcripts = {}
    completed = 0
    
    # Use asyncio.as_completed for real-time progress updates
    for completed_task in asyncio.as_completed(tasks):
        try:
            result = await completed_task
            if isinstance(result, tuple) and len(result) == 2:
                index, text = result
                transcripts[index] = text
                completed += 1
                logger.info(f"✅ Completed {completed}/{total_tasks} segments (Progress: {completed/total_tasks*100:.1f}%)")
            else:
                # Handle unexpected result format
                completed += 1
                logger.info(f"⚠️  Unexpected result format for segment {completed}")
                
        except Exception as e:
            completed += 1
            logger.error(f"❌ Task failed: {e}")
            logger.info(f"❌ Failed {completed}/{total_tasks} segments (Progress: {completed/total_tasks*100:.1f}%)")
    
    logger.info(f"🎉 ASR processing completed! Processed {len(transcripts)} segments successfully.")
    
    return transcripts


async def speech_to_text_async(video_name, working_dir, segment_index2name, audio_output_format, global_config):
    """
    Async speech-to-text function using Alibaba Cloud DashScope online ASR
    
    Args:
        video_name: Name of the video
        working_dir: Working directory
        segment_index2name: Mapping of segment indices to names
        audio_output_format: Audio file format
        global_config: Global configuration dictionary containing API keys and settings
    """
    api_key = global_config.get('ali_dashscope_api_key')
    
    if not api_key:
        raise ValueError("ali_dashscope_api_key must be provided in global_config for online ASR")
    
    return await speech_to_text_online(
        video_name, working_dir, segment_index2name, audio_output_format, global_config
    )

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format, global_config):
    """
    Synchronous wrapper for async speech-to-text function
    
    Args:
        video_name: Name of the video
        working_dir: Working directory
        segment_index2name: Mapping of segment indices to names
        audio_output_format: Audio file format
        global_config: Global configuration dictionary containing API keys and settings
    """
    # Run the async function in an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        speech_to_text_async(video_name, working_dir, segment_index2name, audio_output_format, global_config)
    )