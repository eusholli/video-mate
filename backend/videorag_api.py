# type: ignore
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
import time
import threading
import multiprocessing
import base64
import pickle
import requests
import numpy as np
from typing import List
import socket
import datetime
import json
import signal
import atexit
import psutil
import hashlib
import shutil
import gc
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import logging
import warnings

# --- ADD THIS SECTION HERE ---
import dashscope
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- ⬇️ ADD THIS BLOCK ⬇️ ---
# Force Singapore (International) Endpoints
print("🌍 Forcing International Endpoints...")
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.base_websocket_api_url = (
    "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/"
)
# -----------------------------

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from videorag._llm import (
    LLMConfig,
    openai_embedding,
    gpt_complete,
    dashscope_caption_complete,
)
from videorag import VideoRAG, QueryParam
from videorag.unified_storage import UnifiedNanoVectorDBStorage
from videorag._op import search_precise_clips
import video_clip_api
from video_clip_api import generate_clip_file

# Log recording function
def log_to_file(message, log_file="log.txt"):
    """Log messages to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure the log file is created in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, log_file)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[LOG] {message}")  # Add prefix to distinguish
    except Exception as e:
        print(f"[ERROR] Failed to write to log: {e}")
        print(f"[LOG] {message}")  # At least output to console

# --- Helper Functions ---

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_json_file(file_path, default=[]):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(default, f)
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except:
            return default

def save_json_file(file_path, data):
    """Atomic save to prevent race conditions with readers"""
    temp_path = f"{file_path}.tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno()) # Ensure write to disk
        os.replace(temp_path, file_path) # Atomic switch
    except Exception as e:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        raise e

# --- Managers ---

class GlobalImageBindManager:
    """Global ImageBind manager, providing HTTP API interface"""
    def __init__(self):
        self.embedder = None
        self.is_initialized = False
        self.is_loaded = False
        self.usage_count = 0
        self.model_config = None
        self.model_path = None
        self._lock = threading.Lock()

    def initialize(self, model_path: str):
        with self._lock:
            if self.is_initialized and self.model_path == model_path:
                return
            self.model_path = model_path
            self.model_config = {"model_path": model_path, "configured_at": time.time()}
            self.is_initialized = True
            log_to_file(f"✅ ImageBind manager configured with model path: {model_path}")

    def ensure_imagebind_loaded(self):
        with self._lock:
            if self.is_loaded: return True
            if not self.is_initialized: raise RuntimeError("ImageBind not initialized")
            
            try:
                log_to_file("🔄 Loading ImageBind model...")
                import torch
                from imagebind.models.imagebind_model import ImageBindModel
                from videorag._utils import get_imagebind_device
                from videorag._videoutil import encode_video_segments, encode_string_query

                device = get_imagebind_device()
                self.embedder = ImageBindModel(
                    vision_embed_dim=1280, vision_num_blocks=32, vision_num_heads=16,
                    text_embed_dim=1024, text_num_blocks=24, text_num_heads=16,
                    out_embed_dim=1024, audio_drop_path=0.1, imu_drop_path=0.7,
                )
                
                # OPTIMIZATION: Use mmap=True to avoid loading entire file into RAM at once
                # This helps fit the 4.5GB model into 8GB Docker limits
                log_to_file(f"Memory before load: {psutil.virtual_memory().available / 1024**3:.2f} GB")
                state_dict = torch.load(self.model_path, map_location=device, mmap=True)
                self.embedder.load_state_dict(state_dict)
                del state_dict
                gc.collect()
                
                self.embedder = self.embedder.to(device)
                self.embedder.eval()
                self.is_loaded = True
                log_to_file("✅ ImageBind model loaded successfully")
                return True
            except Exception as e:
                log_to_file(f"❌ Failed to load ImageBind: {str(e)}")
                raise

    def release_imagebind(self):
        with self._lock:
            if not self.is_loaded: return
            self.embedder = None
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.is_loaded = False

    def encode_video_segments(self, video_batch: List[str]) -> np.ndarray:
        with self._lock:
            if not self.is_loaded: raise RuntimeError("ImageBind not loaded")
            self.usage_count += 1
            from videorag._videoutil import encode_video_segments
            return encode_video_segments(video_batch, self.embedder)

    def encode_string_query(self, query: str) -> np.ndarray:
        with self._lock:
            if not self.is_loaded: raise RuntimeError("ImageBind not loaded")
            self.usage_count += 1
            from videorag._videoutil import encode_string_query
            return encode_string_query(query, self.embedder)
            
    def get_status(self) -> dict:
        with self._lock:
            return {"initialized": self.is_initialized, "loaded": self.is_loaded}


class HTTPImageBindClient:
    """HTTP client for subprocesses"""
    def __init__(self, base_url: str = "http://localhost:64451"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def encode_video_segments(self, video_batch: List[str]) -> np.ndarray:
        res = self.session.post(f"{self.base_url}/api/imagebind/encode/video", json={"video_batch": video_batch}, timeout=1800)
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                err_text = res.json().get("error", res.text)
            except:
                err_text = res.text
            log_to_file(f"❌ HTTP Client Error in encode_video_segments: {e}. Status: {res.status_code}. Response: {err_text}")
            raise e
        data = res.json()
        if not data["success"]: raise RuntimeError(data["error"])
        return pickle.loads(base64.b64decode(data["result"]))

    def encode_string_query(self, query: str) -> np.ndarray:
        res = self.session.post(f"{self.base_url}/api/imagebind/encode/query", json={"query": query}, timeout=1800)
        res.raise_for_status()
        data = res.json()
        if not data["success"]: raise RuntimeError(data["error"])
        return pickle.loads(base64.b64decode(data["result"]))
        
    def get_status(self) -> dict:
        try:
             res = self.session.get(f"{self.base_url}/api/imagebind/status", timeout=10)
             return res.json().get("status", {})
        except:
            return {"loaded": False}


class LibraryManager:
    def __init__(self, base_storage_path):
        self.base_path = base_storage_path
        self.library_dir = os.path.join(base_storage_path, "library")
        self.library_json = os.path.join(self.library_dir, "library.json")
        os.makedirs(self.library_dir, exist_ok=True)

    def list_videos(self):
        return ensure_json_file(self.library_json, [])

    def get_video(self, video_id):
        videos = self.list_videos()
        for v in videos:
            if v["id"] == video_id: return v
        return None

    def add_video_entry(self, video_id, title, original_path):
        videos = self.list_videos()
        # Check if exists
        for v in videos:
            if v["id"] == video_id:
                return v # Return existing
        
        new_entry = {
            "id": video_id,
            "title": title,
            "original_path": original_path,
            "status": "processing",
            "progress": 0,
            "phase": "Initializing",
            "created_at": time.time(),
            "updated_at": time.time()
        }
        videos.append(new_entry)
        save_json_file(self.library_json, videos)
        return new_entry

    def update_status(self, video_id, status, error=None, progress=None, phase=None):
        videos = self.list_videos()
        for v in videos:
            if v["id"] == video_id:
                v["status"] = status
                if error: v["error"] = error
                if progress is not None: v["progress"] = progress
                if phase: v["phase"] = phase
                v["updated_at"] = time.time()
                save_json_file(self.library_json, videos)
                return

    def delete_video(self, video_id):
        videos = self.list_videos()
        videos = [v for v in videos if v["id"] != video_id]
        save_json_file(self.library_json, videos)
        
        # Remove dir
        video_dir = os.path.join(self.library_dir, video_id)
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)

class SessionManager:
    def __init__(self, base_storage_path):
        self.sessions_dir = os.path.join(base_storage_path, "sessions")
        self.sessions_json = os.path.join(self.sessions_dir, "sessions.json")
        os.makedirs(self.sessions_dir, exist_ok=True)

    def list_sessions(self):
        return ensure_json_file(self.sessions_json, [])

    def create_session(self, name, video_ids):
        sessions = self.list_sessions()
        # sort list by updated_at
        session_id = str(len(sessions) + 1)
        # Assuming simple ID or uuid
        import uuid
        session_id = str(uuid.uuid4())
        
        new_session = {
            "id": session_id,
            "name": name,
            "video_ids": video_ids,
            "created_at": time.time(),
            "last_active": time.time()
        }
        sessions.append(new_session)
        save_json_file(self.sessions_json, sessions)
        
        # Create session dir
        session_path = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_path, exist_ok=True)
        
        return new_session

    def delete_session(self, session_id):
        sessions = self.list_sessions()
        sessions = [s for s in sessions if s["id"] != session_id]
        save_json_file(self.sessions_json, sessions)
        
        session_path = os.path.join(self.sessions_dir, session_id)
        if os.path.exists(session_path):
            shutil.rmtree(session_path)

    def get_session(self, session_id):
        sessions = self.list_sessions()
        for s in sessions:
            if s["id"] == session_id: return s
        return None

    def update_last_active(self, session_id):
        sessions = self.list_sessions()
        for s in sessions:
            if s["id"] == session_id:
                s["last_active"] = time.time()
                break
        save_json_file(self.sessions_json, sessions)

# --- Global State ---
global_imagebind_manager = GlobalImageBindManager()
library_manager = None
session_manager = None
global_config = None

def get_library_manager():
    global library_manager
    if not library_manager:
        config = load_and_init_global_config()[2] # Hack to get config
        if config: library_manager = LibraryManager(config["base_storage_path"])
    return library_manager

def get_session_manager():
    global session_manager
    if not session_manager:
        config = load_and_init_global_config()[2]
        if config: session_manager = SessionManager(config["base_storage_path"])
    return session_manager

# --- Worker Processes ---

def ingest_worker_process(video_id, file_path, global_config, server_url, resume=False):
    """
    Worker to ingest a single video into library/<video_id>
    """
    import setproctitle
    setproctitle.setproctitle(f"videorag-ingest-{video_id}")
    
    try:
        lib_mgr = LibraryManager(global_config["base_storage_path"])
        working_dir = os.path.join(lib_mgr.library_dir, video_id)
        os.makedirs(working_dir, exist_ok=True)
        
        # Status callback
        def progress_callback(percent, step, msg):
             log_to_file(f"[{video_id}] {percent}% {step}: {msg}")
             try:
                 lib_mgr.update_status(video_id, "processing", progress=percent, phase=step)
             except Exception as e:
                 log_to_file(f"Failed to update progress for {video_id}: {e}")

        # Setup VideoRAG
        imagebind_client = HTTPImageBindClient(server_url)
        
        # Wait for LB
        for i in range(30):
            st = imagebind_client.get_status()
            if st.get("loaded"): break
            time.sleep(1)
            
        videorag_llm_config = LLMConfig(
            embedding_func_raw=openai_embedding,
            embedding_model_name="text-embedding-3-small",
            embedding_dim=1536,
            embedding_max_token_size=8192,
            embedding_batch_num=32,
            embedding_func_max_async=16,
            query_better_than_threshold=0.2,
            best_model_func_raw=gpt_complete,
            best_model_name=global_config.get("analysisModel"),
            best_model_max_token_size=32768,
            best_model_max_async=16,
            cheap_model_func_raw=gpt_complete,
            cheap_model_name=global_config.get("processingModel"),
            cheap_model_max_token_size=32768,
            cheap_model_max_async=16,
            caption_model_func_raw=dashscope_caption_complete,
            caption_model_name=global_config.get("caption_model"),
            caption_model_max_async=3,
        )
        
        rag = VideoRAG(
            llm=videorag_llm_config,
            working_dir=working_dir,
            ali_dashscope_api_key=global_config.get("ali_dashscope_api_key"),
            ali_dashscope_base_url=global_config.get("ali_dashscope_base_url"),
            caption_model=global_config.get("caption_model"),
            asr_model=global_config.get("asr_model"),
            openai_api_key=global_config.get("openai_api_key"),
            openai_base_url=global_config.get("openai_base_url"),
            imagebind_client=imagebind_client,
        )
        
        # Rename file to <id>.mp4/etc to ensure consistent naming inside VideoRAG
        # copy file to working dir
        ext = os.path.splitext(file_path)[1]
        target_path = os.path.join(working_dir, f"{video_id}{ext}")
        if not os.path.exists(target_path):
            shutil.copy2(file_path, target_path)
            
        rag.insert_video(
            video_path_list=[target_path],
            progress_callback=progress_callback,
            resume=resume
        )
        
        lib_mgr.update_status(video_id, "ready", progress=100, phase="Ready")
        log_to_file(f"✅ Ingestion complete for {video_id}")
        
    except Exception as e:
        log_to_file(f"❌ Ingestion failed for {video_id}: {e}")
        lib_mgr = LibraryManager(global_config["base_storage_path"])
        lib_mgr.update_status(video_id, "error", str(e))

def setup_session_rag(session_id, global_config, server_url, force_rebuild=False):
    """
    Setup VideoRAG instance for a session (Merging DBs if needed).
    """
    sess_mgr = SessionManager(global_config["base_storage_path"])
    lib_mgr = LibraryManager(global_config["base_storage_path"])
    
    session = sess_mgr.get_session(session_id)
    if not session: raise ValueError("Session not found")
    
    session_dir = os.path.join(sess_mgr.sessions_dir, session_id)
    
    # Check if we need to rebuild (if forced or missing files)
    chunks_vdb_path = os.path.join(session_dir, "vdb_chunks.json")
    if force_rebuild or not os.path.exists(chunks_vdb_path):
        log_to_file(f"🔨 Rebuilding session DBs for {session_id}")
        
        # 1. Chunks
        chunks_paths = [os.path.join(lib_mgr.library_dir, vid, "vdb_chunks.json") for vid in session["video_ids"]]
        
        # 2. Entities
        entities_paths = [os.path.join(lib_mgr.library_dir, vid, "vdb_entities.json") for vid in session["video_ids"]]
        
        # 3. Video Features
        features_paths = [os.path.join(lib_mgr.library_dir, vid, "vdb_video_segment_feature.json") for vid in session["video_ids"]]
        
        from nano_vectordb import NanoVectorDB
        
        def merge_vdb(source_paths, target_path, dim):
            # Create/Reset target
            client = NanoVectorDB(dim, storage_file=target_path)
            data_buffer = []
            
            for sp in source_paths:
                if os.path.exists(sp):
                    try:
                        source_client = NanoVectorDB(dim, storage_file=sp)
                        storage = getattr(source_client, '_NanoVectorDB__storage', {})
                        if isinstance(storage, dict) and "data" in storage and "matrix" in storage:
                            items = storage["data"]
                            matrix = storage["matrix"]
                            if len(matrix) >= len(items):
                                for i, item in enumerate(items):
                                    new_item = item.copy() 
                                    new_item["__vector__"] = matrix[i]
                                    data_buffer.append(new_item)
                    except Exception as e:
                        log_to_file(f"Failed to merge VDB {sp}: {e}")

            if data_buffer:
                client.upsert(data_buffer)
            client.save()
            
        merge_vdb(chunks_paths, os.path.join(session_dir, "vdb_chunks.json"), 1536)
        merge_vdb(entities_paths, os.path.join(session_dir, "vdb_entities.json"), 1536)
        merge_vdb(features_paths, os.path.join(session_dir, "vdb_video_segment_feature.json"), 1024)
        
        def merge_kv(source_paths, target_path):
             merged = {}
             for sp in source_paths:
                 if os.path.exists(sp):
                     with open(sp, 'r') as f:
                        d = json.load(f)
                        merged.update(d)
             with open(target_path, 'w') as f:
                 json.dump(merged, f)
                 
        kv_chunks_paths = [os.path.join(lib_mgr.library_dir, vid, "kv_store_text_chunks.json") for vid in session["video_ids"]]
        merge_kv(kv_chunks_paths, os.path.join(session_dir, "kv_store_text_chunks.json"))
        
        kv_segs_paths = [os.path.join(lib_mgr.library_dir, vid, "kv_store_video_segments.json") for vid in session["video_ids"]]
        merge_kv(kv_segs_paths, os.path.join(session_dir, "kv_store_video_segments.json"))
        
        kv_paths_paths = [os.path.join(lib_mgr.library_dir, vid, "kv_store_video_path.json") for vid in session["video_ids"]]
        merge_kv(kv_paths_paths, os.path.join(session_dir, "kv_store_video_path.json"))
        
        # Merge Graph
        try:
            import networkx as nx
            G = nx.Graph()
            for vid in session["video_ids"]:
                gp = os.path.join(lib_mgr.library_dir, vid, "graph_chunk_entity_relation.graphml")
                if os.path.exists(gp):
                    g_sub = nx.read_graphml(gp)
                    G = nx.compose(G, g_sub)
            nx.write_graphml(G, os.path.join(session_dir, "graph_chunk_entity_relation.graphml"))
        except Exception as e:
            log_to_file(f"Graph merge failed: {e}")

    # SETUP RAG INSTANCE
    imagebind_client = HTTPImageBindClient(server_url)
    videorag_llm_config = LLMConfig(
        embedding_func_raw=openai_embedding,
        embedding_model_name="text-embedding-3-small",
        embedding_dim=1536,
        embedding_max_token_size=8192,
        embedding_batch_num=32,
        embedding_func_max_async=16,
        query_better_than_threshold=0.2,
        best_model_func_raw=gpt_complete,
        best_model_name=global_config.get("analysisModel"),
        best_model_max_token_size=32768,
        best_model_max_async=16,
        cheap_model_func_raw=gpt_complete,
        cheap_model_name=global_config.get("processingModel"),
        cheap_model_max_token_size=32768,
        cheap_model_max_async=16,
        caption_model_func_raw=dashscope_caption_complete,
        caption_model_name=global_config.get("caption_model"),
        caption_model_max_async=3,
    )
    
    # Collect Video Titles for display
    def clean_title(v):
        original = v.get("title", os.path.basename(v.get("original_path", "Unknown")))
        import re
        match = re.match(r'^\d+_(.*)$', original)
        if match: return match.group(1)
        return original

    video_titles = {v["id"]: clean_title(v) for v in lib_mgr.list_videos()}

    rag = VideoRAG(
        llm=videorag_llm_config,
        working_dir=session_dir,
        ali_dashscope_api_key=global_config.get("ali_dashscope_api_key"),
        ali_dashscope_base_url=global_config.get("ali_dashscope_base_url"),
        caption_model=global_config.get("caption_model"),
        asr_model=global_config.get("asr_model"),
        openai_api_key=global_config.get("openai_api_key"),
        openai_base_url=global_config.get("openai_base_url"),
        imagebind_client=imagebind_client,
        addon_params={"server_url": server_url, "video_titles": video_titles},
    )
    return rag

def query_worker_process(session_id, query, global_config, server_url):
    """
    Worker for querying.
    """
    import setproctitle
    setproctitle.setproctitle(f"videorag-query-{session_id}")
    
    try:
        sess_mgr = SessionManager(global_config["base_storage_path"])
        session_dir = os.path.join(sess_mgr.sessions_dir, session_id)
        
        rag = setup_session_rag(session_id, global_config, server_url, force_rebuild=True)
        
        param = QueryParam(mode="videorag")
        param.wo_reference = True
        response = rag.query(query=query, param=param)
        if isinstance(response, tuple):
            response_text, sources = response
        else:
            response_text = response
            sources = []
        
        # Sanitize for JSON
        def make_serializable(obj):
            import numpy as np
            if isinstance(obj, np.floating):
                v = float(obj)
                if np.isnan(v) or np.isinf(v): return None
                return v
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return make_serializable(obj.tolist())
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            if hasattr(obj, "to_dict"):
                return make_serializable(obj.to_dict())
            if hasattr(obj, "__dict__"):
                return make_serializable(obj.__dict__)
            return obj

        sources = make_serializable(sources)
        
        # Resolve video_ids for sources
        try:
            lm = get_library_manager()
            videos = lm.list_videos()
            # RAG uses filename stem as video_name, which matches default title
            # RAG uses filename stem as video_name.
            # In our case, video_name often equals the video_id (MD5 hash).
            name_to_id = {v['title']: v['id'] for v in videos}
            valid_ids = {v['id'] for v in videos}
            
            for s in sources:
                if isinstance(s, dict) and "video_name" in s:
                    vn = s["video_name"]
                    
                    # 1. Direct match with ID (most likely)
                    if vn in valid_ids:
                         s["video_id"] = vn
                    # 2. Match with Title
                    elif vn in name_to_id:
                        s["video_id"] = name_to_id[vn]
                    else:
                        # 3. Fallback: try case-insensitive title match
                        for k, v_id in name_to_id.items():
                            if k.lower() == vn.lower():
                                s["video_id"] = v_id
                                break
        except Exception as e:
            print(f"Error resolving source video_ids: {e}")

        # History Handling
        history_file = os.path.join(session_dir, "history.json")
        history = ensure_json_file(history_file, [])
        history.append({
            "role": "assistant", 
            "content": response_text, 
            "sources": sources,
            "timestamp": datetime.datetime.now().isoformat()
        })
        save_json_file(history_file, history)
        
        # Update Status
        status_file = os.path.join(session_dir, "status.json")
        res = {"query_status": {"status": "completed", "answer": response_text, "sources": sources, "query": query}}
        save_json_file(status_file, res)
        
        # Update Session Last Active
        try:
            sm = SessionManager(global_config["base_storage_path"])
            # reload to get fresh
            sessions = sm.list_sessions()
            for s in sessions:
                if s["id"] == session_id:
                    s["last_active"] = time.time()
                    break
            save_json_file(sm.sessions_json, sessions)
        except Exception: pass
        
    except Exception as e:
        log_to_file(f"Query Process Failed: {e}")
        try:
            status_file = os.path.join(os.path.join(sess_mgr.sessions_dir, session_id), "status.json")
            save_json_file(status_file, {"query_status": {"status": "error", "message": str(e)}})
        except: pass

def clip_generator_worker(session_id, query, global_config, server_url):
    """
    Worker to find and generate clips.
    """
    import asyncio
    try:
        log_to_file(f"Generating clips for session {session_id}, query: {query}")
        # Reuse existing DBs (no force rebuild)
        rag = setup_session_rag(session_id, global_config, server_url, force_rebuild=False)
        
        # Run search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        clips = loop.run_until_complete(search_precise_clips(
            query,
            rag.entities_vdb,
            rag.text_chunks,
            rag.chunks_vdb,
            rag.video_segments,
            rag.video_segment_feature_vdb,
            rag.chunk_entity_relation_graph,
            QueryParam(mode="videorag"),
            rag.safe_config
        ))
        
        # Generate Clip Files
        generated_clips = []
        lib_mgr = LibraryManager(global_config["base_storage_path"])
        
        for clip in clips:
            video_id = clip["video_id"]
            start = clip["start"]
            end = clip["end"]
            
            # Output path: library/<vid>/clips/<start-end>.mp4
            clips_dir = os.path.join(lib_mgr.library_dir, video_id, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            output_name = f"clip_{int(start*1000)}_{int(end*1000)}.mp4"
            output_path = os.path.join(clips_dir, output_name)
            
            video_entry = lib_mgr.get_video(video_id)
            if not video_entry: continue
            
            original_path = video_entry.get("original_path")
            # Or use working dir path?
            # Ideally use the one in working dir (rag knows it?)
            # rag.video_path_db stores it?
            # Let's fallback to original_path
            
            # Check if exists
            if not os.path.exists(output_path):
                # Generate
                success = generate_clip_file(original_path, start, end, output_path)
                if not success: continue
            
            # Add URL
            # URL: /api/library/<vid>/clips/<output_name>
            clip["url"] = f"{server_url}/api/library/{video_id}/clips/{output_name}"
            # Add Title/Thumb?
            clip["title"] = video_entry.get("title", "Video")
            generated_clips.append(clip)
            
        # Update Session with these clips?
        # Or just return? The endpoint is async polled?
        # The user wants "return a list of clips". 
        # API is POST -> returns Job ID? Or waits?
        # If we wait, it might timeout.
        # Let's write to a status file for "clip_generation"? or append to history?
        # Request says: "The above replaces the existing clip return service."
        # "button... triggers the transcript searching, generating, and returning..."
        
        # If I want to return them to frontend, I should probably save them to a file that frontend polls?
        # OR, since we are using `querySession` which polls `status.json`.
        # I can reuse `status.json` structure?
        # `status.json` has `query_status`.
        # I can add `clip_status`.
        
        sess_mgr = SessionManager(global_config["base_storage_path"])
        session_dir = os.path.join(sess_mgr.sessions_dir, session_id)
        status_file = os.path.join(session_dir, "status.json")
        
        # Read existing to preserve query_status?
        current = ensure_json_file(status_file, {})
        current["clip_status"] = {
            "status": "completed",
            # We don't need to store clips here anymore since they go to history
            # But keeping them for now doesn't hurt, helps debugging
            "clips": generated_clips 
        }
        save_json_file(status_file, current)
        
        # --- Persist to History ---
        history_path = os.path.join(session_dir, "history.json")
        try:
             import datetime
             history = ensure_json_file(history_path, [])
             history.append({
                 "role": "assistant", 
                 "content": "Here are the relevant clips based on your request:", 
                 "clips": generated_clips,
                 "timestamp": datetime.datetime.now().isoformat()
             })
             with open(history_path, 'w') as f:
                 import json
                 json.dump(history, f, indent=2)
        except Exception as e:
            log_to_file(f"Failed to append clips to history: {e}")
        
    except Exception as e:
        log_to_file(f"Clip generation failed: {e}")
        # Write error
        try:
            sess_mgr = SessionManager(global_config["base_storage_path"])
            session_dir = os.path.join(sess_mgr.sessions_dir, session_id)
            status_file = os.path.join(session_dir, "status.json")
            current = ensure_json_file(status_file, {})
            current["clip_status"] = {"status": "error", "message": str(e)}
            save_json_file(status_file, current)
        except: pass

# --- FLASK ENDPOINTS FOR CLIPS ---
pass # Handled in create_app modification next


# --- Flask App ---

def load_and_init_global_config():
    config = {
        "ali_dashscope_api_key": os.getenv("DASHSCOPE_API_KEY"),
        "ali_dashscope_base_url": os.getenv("DASHSCOPE_BASE_URL"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "caption_model": os.getenv("CAPTION_MODEL"),
        "asr_model": os.getenv("ASR_MODEL"),
        "analysisModel": os.getenv("ANALYSIS_MODEL"),
        "processingModel": os.getenv("PROCESSING_MODEL"),
        "base_storage_path": os.getenv("BASE_STORAGE_PATH"),
        "image_bind_model_path": os.getenv("IMAGE_BIND_MODEL_PATH", "/app/models/imagebind_huge.pth"),
    }
    # Init ImageBind
    if config["image_bind_model_path"]:
        global_imagebind_manager.initialize(config["image_bind_model_path"])
        # Auto-load to ensure readiness immediately (prevents 503s on startup)
        global_imagebind_manager.ensure_imagebind_loaded() 
    return True, "Config Loaded", config

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Load config on startup
    global global_config
    _, _, global_config = load_and_init_global_config()
    
    # --- NEW ENDPOINTS ---
    

    @app.route("/api/library/<video_id>/clip", methods=["POST"])
    def generate_manual_clip(video_id):
        """Generate a clip from start to end timestamp manually"""
        try:
            data = request.json
            start = data.get("start")
            end = data.get("end")
            
            if start is None or end is None:
                return jsonify({"success": False, "error": "Missing start/end"}), 400
                
            lib_mgr = get_library_manager()
            video_entry = lib_mgr.get_video(video_id)
            if not video_entry:
                return jsonify({"success": False, "error": "Video not found"}), 404
                
            # Output path
            clips_dir = os.path.join(lib_mgr.library_dir, video_id, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            output_name = f"manual_{int(start*1000)}_{int(end*1000)}.mp4"
            output_path = os.path.join(clips_dir, output_name)
            
            # Check if exists
            if not os.path.exists(output_path):
                 success = generate_clip_file(video_entry["original_path"], start, end, output_path)
                 if not success:
                     return jsonify({"success": False, "error": "Clip generation failed"}), 500
                     
            server_url = f"http://localhost:{globals().get('SERVER_PORT', 64451)}"
            url = f"{server_url}/api/library/{video_id}/clips/{output_name}"
            return jsonify({
                "success": True, 
                "clip": {
                    "url": url,
                    "start": start,
                    "end": end,
                    "title": video_entry.get("title", "Video"),
                    "video_id": video_id
                }
            })

        except Exception as e:
            log_to_file(f"Manual clip generation failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
            
    # --- END NEW ENDPOINTS ---


    @app.route("/api/library/<video_id>/stream", methods=["GET"])
    def stream_video(video_id):
        """Stream the video file"""
        try:
            lib_mgr = get_library_manager()
            video_entry = lib_mgr.get_video(video_id)
            if not video_entry:
                return jsonify({"success": False, "error": "Video not found"}), 404
            
            # Prefer using the file in working_dir (copied during ingest) to ensure existence?
            # Or original path?
            # Original path might reference temp upload which might be gone?
            # Ingest copies it to working_dir/<id>.ext
            
            working_dir = os.path.join(lib_mgr.library_dir, video_id)
            # Find video file in working_dir
            # It should be <video_id>.* 
            
            target_file = None
            if os.path.exists(working_dir):
                for f in os.listdir(working_dir):
                    if f.startswith(video_id + "."):
                        target_file = os.path.join(working_dir, f)
                        break
            
            if not target_file:
                # Fallback to original path
                 if os.path.exists(video_entry["original_path"]):
                     target_file = video_entry["original_path"]
            
            if not target_file or not os.path.exists(target_file):
                return jsonify({"success": False, "error": "Video file not found"}), 404
                
            return send_file(target_file) 
            # Note: For production streaming, need range support. 
            # Flask send_file supports it basically, but for large files it's not ideal.
            # But for this prototype, it's sufficient.
            
        except Exception as e:
            import traceback
            log_to_file(f"❌ Error streaming video {video_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"success": False, "error": str(e)}), 500
            

        return jsonify({"status": "ok"})
        
    @app.route("/api/initialize", methods=["POST"])
    def initialize():
        s, m, _ = load_and_init_global_config()
        return jsonify({"success": s, "message": m})

    # --- Library Endpoints ---
    @app.route("/api/health", methods=["GET"])
    def health_check():
        lm = get_library_manager()
        return jsonify({"success": True, "videos": lm.list_videos()})
        
    @app.route("/api/library", methods=["GET"])
    def list_library():
        lm = get_library_manager()
        return jsonify({"success": True, "videos": lm.list_videos()})
        
    @app.route("/api/library/ingest", methods=["POST"])
    def ingest_video():
        data = request.json
        path = data.get("path")
        resume = data.get("resume", True) # Default to True for UX
        if not path or not os.path.exists(path):
            return jsonify({"success": False, "error": "Invalid path"}), 400
            
        lm = get_library_manager()
        
        # Calculate ID
        video_id = calculate_md5(path)
        title = os.path.basename(path)
        
        # Create entry in library (this ensures it shows up in the UI)
        lm.add_video_entry(video_id, title, path)
        
        success, msg, global_config = load_and_init_global_config()
        
        # Start worker
        # We need to determine the server URL for the worker to call back (ImageBind)
        # We'll use local loopback on the port we know we are running on (64451 from logs)
        server_url = "http://127.0.0.1:64451"
        
        p = multiprocessing.Process(target=ingest_worker_process, args=(video_id, path, global_config, server_url, resume))
        p.start()
        
        return jsonify({"success": True, "video_id": video_id, "message": "Ingestion started"})

    @app.route("/api/library/<video_id>/clips/<filename>", methods=["GET"])
    def get_generated_clip(video_id, filename):
        lm = get_library_manager()
        clip_path = os.path.join(lm.library_dir, video_id, "clips", filename)
        if os.path.exists(clip_path):
            return send_file(clip_path, as_attachment=True, download_name=filename)
        return jsonify({"success": False, "error": "Clip not found"}), 404

    @app.route("/api/library/<video_id>/transcript", methods=["GET"])
    def get_transcript(video_id):
        lm = get_library_manager()
        video_entry = lm.get_video(video_id)
        if not video_entry:
            return jsonify({"success": False, "error": "Video not found"}), 404
        
        # Access KV Store directly
        kv_path = os.path.join(lm.library_dir, video_id, "kv_store_video_segments.json")
        if not os.path.exists(kv_path):
             return jsonify({"success": False, "error": "Transcript not available"}), 404
             
        try:
            with open(kv_path, 'r') as f:
                segments_data = json.load(f)
            
            # Handle nested structure {video_id: {index: data}} vs flat {video_id_index: data}
            target_segments = {}
            if video_id in segments_data:
                target_segments = segments_data[video_id]
            else:
                target_segments = segments_data

            sorted_segments = []
            for key, val in target_segments.items():
                try:
                    # Key could be "0" (nested) or "vid_0" (flat)
                    if "_" in key:
                         if not key.startswith(video_id): continue
                         idx = int(key.split("_")[-1])
                    else:
                         idx = int(key)

                    val["id"] = f"{video_id}_{idx}" # Normalizing ID
                    val["index"] = idx
                    
                    # Parse time string "start-end"
                    if "time" in val:
                        start, end = val["time"].split("-")
                        val["start_time"] = float(start)
                        val["end_time"] = float(end)
                    elif "start" in val and "end" in val:
                        # Sometimes it might be pre-parsed?
                         val["start_time"] = float(val["start"])
                         val["end_time"] = float(val["end"])
                    
                    # Fix Relative Timestamps in sentences
                    # The 'sentences' generated by the tool are often relative to the chunk start
                    if "detailed_transcript" in val:
                        dt = val["detailed_transcript"]
                        offset_ms = val["start_time"] * 1000
                        
                        if "sentences" in dt:
                            for s in dt["sentences"]:
                                s["start"] = s["start"] + offset_ms
                                s["end"] = s["end"] + offset_ms
                                
                        if "words" in dt:
                            for w in dt["words"]:
                                w["start"] = w["start"] + offset_ms
                                w["end"] = w["end"] + offset_ms

                    sorted_segments.append(val)
                except: continue
                
            sorted_segments.sort(key=lambda x: x["index"])
            
            # Flatten to sentences for frontend
            full_transcript = []
            for seg in sorted_segments:
                if "detailed_transcript" in seg and "sentences" in seg["detailed_transcript"]:
                    # Already offset-adjusted above
                    # Ensure words are propagated if strictly needed by frontend, usually they are attached to sentences?
                    # In videorag 'sentences' might strictly contain text/start/end.
                    # 'words' are usually a separate list in detailed_transcript or attached to sentences?
                    # Let's check how 'offset_ms' logic was applied.
                    # It iterated `dt["sentences"]` AND `dt["words"]`.
                    # So `words` is a sibling of `sentences` in `detailed_transcript`.
                    # We should probably return a structure that supports both.
                    # But the frontend loop iterates `sentences`.
                    # We should attach the relevant words TO the sentence?
                    # Or just return a flat list of sentences, and each sentence has a `words` property?
                    # The current logic extends `full_transcript` with `dt["sentences"]`.
                    # If `dt["sentences"]` items don't have `words`, we lose them.
                    
                    # Heuristic: Filter main `dt["words"]` into these sentences based on time range?
                    # This is expensive. 
                    # Does the ASR tool output words *inside* sentences?
                    # Dashscope/Whisper outputs might differ.
                    # If we look at lines 1005-1008:
                    # if "words" in dt: for w in dt["words"]: adjust...
                    
                    # So `words` are in `dt`.
                    # We need to assign them to sentences.
                    sentences = seg["detailed_transcript"]["sentences"]
                    words = seg["detailed_transcript"].get("words", [])
                    
                    for s in sentences:
                        s_start = s["start"]
                        s_end = s["end"]
                        # Find words in this range
                        s["words"] = [w for w in words if w["start"] >= s_start and w["end"] <= s_end]
                        
                    full_transcript.extend(sentences)
                else:
                    # Fallback to chunk text
                    full_transcript.append({
                        "text": seg.get("text", ""),
                        "start": seg["start_time"] * 1000,
                        "end": seg["end_time"] * 1000,
                        "words": [] # No words available
                    })
                    
            return jsonify({"success": True, "transcript": full_transcript})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/clip/generate_exact", methods=["POST"])
    def generate_exact_clip():
        data = request.json
        video_id = data.get("video_id")
        start = data.get("start")
        end = data.get("end")
        
        if not video_id or start is None or end is None:
             return jsonify({"success": False, "error": "Missing parameters"}), 400
             
        lm = get_library_manager()
        video_entry = lm.get_video(video_id)
        if not video_entry:
            return jsonify({"success": False, "error": "Video not found"}), 404
            
        original_path = video_entry.get("original_path")
        if not original_path or not os.path.exists(original_path):
             # Try working dir fallback?
             working_path = os.path.join(lm.library_dir, video_id, f"{video_id}{os.path.splitext(original_path or '')[1]}")
             if os.path.exists(working_path):
                 original_path = working_path
             else:
                 return jsonify({"success": False, "error": "Source file not found"}), 404

        # Generate output path
        clips_dir = os.path.join(lm.library_dir, video_id, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        filename = f"exact_{int(start*1000)}_{int(end*1000)}.mp4"
        output_path = os.path.join(clips_dir, filename)
        
        # Check cache
        if not os.path.exists(output_path):
            success = generate_clip_file(original_path, start, end, output_path)
            if not success:
               return jsonify({"success": False, "error": "Generation failed"}), 500
               
        url = f"/api/library/{video_id}/clips/{filename}"
        return jsonify({"success": True, "url": url})

        filename = os.path.basename(path)
        
        # Add to Library (or get existing)
        entry = lm.add_video_entry(video_id, filename, path)
        
        if entry['status'] == 'ready' and not resume:
             # If strictly not resuming and already ready, maybe return?
             # But usually ingestion is idempotent-ish with resume=True.
             pass

        # Start Process
        success, _, config = load_and_init_global_config()
        server_url = f"http://localhost:{globals().get('SERVER_PORT', 64451)}"
        
        p = multiprocessing.Process(target=ingest_worker_process, args=(video_id, path, config, server_url, resume))
        p.start()
        
        return jsonify({"success": True, "status": "processing", "video_id": video_id})

    @app.route("/api/library/upload", methods=["POST"])
    def upload_video():
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400
            
        if file:
            _, _, config = load_and_init_global_config()
            base_dir = config["base_storage_path"]
            upload_dir = os.path.join(base_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            filename = file.filename
            safe_name = f"{int(time.time())}_{filename}"
            save_path = os.path.join(upload_dir, safe_name)
            
            file.save(save_path)
            
            return jsonify({
                "success": True, 
                "path": os.path.abspath(save_path),
                "filename": filename
            })
        
    @app.route("/api/library/<video_id>", methods=["DELETE"])
    def delete_video(video_id):
        lm = get_library_manager()
        lm.delete_video(video_id)
        return jsonify({"success": True})

    @app.route("/api/library/videos/<video_id>/file", methods=["GET"])
    def serve_video_file(video_id):
        lm = get_library_manager()
        video = lm.get_video(video_id)
        if not video:
             return jsonify({"success": False, "error": "Video not found"}), 404
             
        # video["original_path"] might be absolute or relative?
        # Assuming absolute based on ingest.
        # However, for safety and "library" concept, we should probably serve from library dir if we copied it there?
        # In ingest_worker_process:
        # target_path = os.path.join(working_dir, f"{video_id}{ext}")
        # shutil.copy2(file_path, target_path)
        
        # So the video IS in the library dir: library/<video_id>/<video_id>.<ext>
        # Let's find it.
        lib_dir = os.path.join(lm.library_dir, video_id)
        # We don't know the extension easily unless we store it or look for it.
        # We can look for files starting with video_id in that dir.
        
        try:
            files = os.listdir(lib_dir)
            target_file = None
            for f in files:
                if f.startswith(video_id) and f != "vdb_chunks.json": # simple check, better: check extensions
                    # Check if it is a video file?
                    lower = f.lower()
                    if lower.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                        target_file = os.path.join(lib_dir, f)
                        break
            
            if not target_file:
                 # Fallback to original path if still exists?
                 if os.path.exists(video["original_path"]):
                     return send_file(video["original_path"])
                 return jsonify({"success": False, "error": "Video file not found in library"}), 404
                 
            return send_file(target_file)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500



    # --- Session Endpoints ---
    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        sm = get_session_manager()
        return jsonify({"success": True, "sessions": sm.list_sessions()})
        
    @app.route("/api/sessions", methods=["POST"])
    def create_session():
        data = request.json
        name = data.get("name")
        video_ids = data.get("video_ids", [])
        sm = get_session_manager()
        session = sm.create_session(name, video_ids)
        return jsonify({"success": True, "session": session})
        
    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        sm = get_session_manager()
        sm.delete_session(session_id)
        return jsonify({"success": True})
        
    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def get_session_details(session_id):
        sm = get_session_manager()
        
        session = sm.get_session(session_id)
        if not session:
            return jsonify({"success": False, "error": "Not found"}), 404
            
        # Get History
        history_path = os.path.join(sm.sessions_dir, session_id, "history.json")
        history = ensure_json_file(history_path, [])
        
        # Get Status
        status_path = os.path.join(sm.sessions_dir, session_id, "status.json")
        status = ensure_json_file(status_path, {})
        
        return jsonify({
            "success": True, 
            "session": session, 
            "history": history, 
            "status": status.get("query_status"),
            "clip_status": status.get("clip_status")
        })

    @app.route("/api/sessions/<session_id>/query", methods=["POST"])
    def query_session(session_id):
        data = request.json
        query = data.get("query")
        
        _, _, config = load_and_init_global_config()
        server_url = f"http://localhost:{globals().get('SERVER_PORT', 64451)}"
        
        # 1. Update History synchronously (Main Thread)
        sm = get_session_manager()
        history_path = os.path.join(sm.sessions_dir, session_id, "history.json")
        try:
             history = ensure_json_file(history_path, [])
             history.append({
                 "role": "user", 
                 "content": query, 
                 "timestamp": datetime.datetime.now().isoformat()
             })
             with open(history_path, 'w') as f:
                 json.dump(history, f, indent=2)
                 
             # Update Last Active
             sm.update_last_active(session_id)
             
             # Update Status to Processing immediately
             status_path = os.path.join(sm.sessions_dir, session_id, "status.json")
             # Read existing to preserve clip_status if any
             current_status = ensure_json_file(status_path, {})
             current_status["query_status"] = {"status": "processing"}
             save_json_file(status_path, current_status)
             
        except Exception as e:
             log_to_file(f"Failed to sync save history/status: {e}")
        
        # 2. Spawn Worker for AI Response
        p = multiprocessing.Process(target=query_worker_process, args=(session_id, query, config, server_url))
        p.start()
        
        return jsonify({"success": True, "status": "processing"})

    @app.route("/api/sessions/<session_id>/clips", methods=["POST"])
    def generate_session_clips(session_id):
        data = request.json
        query = data.get("query") # Use the query the user wants clips for
        
        _, _, config = load_and_init_global_config()
        server_url = f"http://localhost:{globals().get('SERVER_PORT', 64451)}"
        
        sm = get_session_manager()
        status_path = os.path.join(sm.sessions_dir, session_id, "status.json")
        history_path = os.path.join(sm.sessions_dir, session_id, "history.json")
        
        # Strategy 1: Use the previously generated answer as context
        try:
             history = ensure_json_file(history_path, [])
             # Find last assistant message
             last_answer = None
             for msg in reversed(history):
                 if msg.get("role") == "assistant":
                     last_answer = msg.get("content")
                     break
             
             if last_answer:
                 # Check if it's a string (it should be)
                 if isinstance(last_answer, str):
                     # Limit length to avoid token limits? 
                     # Let's truncate reasonably, e.g. 1000 chars or just pass it.
                     # The answer is usually concise.
                     query = f"{query}\n\nContext from Answer:\n{last_answer}"
                     log_to_file(f"Enriched query with previous answer length: {len(last_answer)}")
        except Exception as e:
            log_to_file(f"Failed to fetch history for context: {e}")

        # Update Status
        try:
            current = ensure_json_file(status_path, {})
            current["clip_status"] = {"status": "processing"}
            save_json_file(status_path, current)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

        # Spawn Worker
        p = multiprocessing.Process(target=clip_generator_worker, args=(session_id, query, config, server_url))
        p.start()
        
        return jsonify({"success": True, "status": "processing"})
        
    @app.route("/api/sessions/<session_id>/status", methods=["GET"])
    def get_query_status(session_id):
        sm = get_session_manager()
        p = os.path.join(sm.sessions_dir, session_id, "status.json")
        if os.path.exists(p):
             with open(p, 'r') as f:
                 st = json.load(f)
                 return jsonify({
                     "success": True, 
                     "status": st.get("query_status"),
                     "clip_status": st.get("clip_status")
                 })
        return jsonify({"success": False, "status": "unknown"})

    # --- ImageBind Internal ---
    @app.route("/api/imagebind/status", methods=["GET"])
    def ib_status():
        status = global_imagebind_manager.get_status()
        if not status.get("loaded"):
            return jsonify({"success": True, "status": status}), 503
        return jsonify({"success": True, "status": status})
        
    @app.route("/api/imagebind/encode/video", methods=["POST"])
    def ib_encode_video():
        try:
             res = global_imagebind_manager.encode_video_segments(request.json["video_batch"])
             return jsonify({"success": True, "result": base64.b64encode(pickle.dumps(res)).decode()})
        except Exception as e:
            import traceback
            error_msg = f"❌ ImageBind Encode Video Failed: {e}\n{traceback.format_exc()}"
            log_to_file(error_msg)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/imagebind/load", methods=["POST"])
    def ib_load():
        try:
            global_imagebind_manager.ensure_imagebind_loaded()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/imagebind/encode/query", methods=["POST"])
    def ib_encode_query():
        try:
             res = global_imagebind_manager.encode_string_query(request.json.get("query"))
             return jsonify({"success": True, "result": base64.b64encode(pickle.dumps(res)).decode()})
        except Exception as e:
            import traceback
            log_to_file(f"❌ ImageBind Encode Query Failed: {e}\n{traceback.format_exc()}")
            return jsonify({"success": False, "error": str(e)}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 64451))
    globals()['SERVER_PORT'] = port 
    app.run(host="0.0.0.0", port=port, threaded=True)
