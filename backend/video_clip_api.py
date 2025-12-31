
import os
import time
import shutil
from flask import Blueprint, request, jsonify, send_file, Response, current_app
from moviepy.video.io.VideoFileClip import VideoFileClip
import threading

# Use a blueprint, but we'll likely just import functions if we are not restructuring the app completely
# For consistency with videorag_api.py, we might just define functions to be registered there.
# But keeping it modular is better.

clip_bp = Blueprint('clip_bp', __name__)

# Cache for generated clips
CLIP_CACHE_DIR = "_clips_cache"

def ensure_cache_dir(base_path):
    path = os.path.join(base_path, CLIP_CACHE_DIR)
    os.makedirs(path, exist_ok=True)
    return path

def generate_clip_file(video_path, start_time, end_time, output_path):
    """
    Generate a clip using MoviePy.
    """
    try:
        # Use a lock-file or similar if concurrency is high? 
        # For MVP, just process.
        with VideoFileClip(video_path) as video:
            # Clamp end time
            if end_time > video.duration:
                end_time = video.duration
            
            subclip = video.subclip(start_time, end_time)
            # Write using appropriate codec
            # 'libx264' is standard. 'aac' for audio.
            subclip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac', 
                verbose=False, 
                logger=None,
                preset='ultrafast' # Optimize for speed
            )
        return True
    except Exception as e:
        print(f"Clip generation failed: {e}")
        return False

@clip_bp.route("/api/clip/generate", methods=["POST"])
def api_generate_clip():
    """
    Endpoint (Optional, valid if Blueprint used)
    """
    pass # Managed in videorag_api for now due to shared 'config' state
