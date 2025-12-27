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
from flask import Flask, request, jsonify
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
                self.embedder.load_state_dict(torch.load(self.model_path, map_location=device))
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
        res.raise_for_status()
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
            "created_at": time.time(),
            "updated_at": time.time()
        }
        videos.append(new_entry)
        save_json_file(self.library_json, videos)
        return new_entry

    def update_status(self, video_id, status, error=None):
        videos = self.list_videos()
        for v in videos:
            if v["id"] == video_id:
                v["status"] = status
                if error: v["error"] = error
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
        def progress_callback(step, msg):
             # We could update a detailed status file in working_dir if needed
             # For now, we rely on 'status.json' managed by split/resume logic
             log_to_file(f"[{video_id}] {step}: {msg}")

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
        
        lib_mgr.update_status(video_id, "ready")
        log_to_file(f"✅ Ingestion complete for {video_id}")
        
    except Exception as e:
        log_to_file(f"❌ Ingestion failed for {video_id}: {e}")
        lib_mgr = LibraryManager(global_config["base_storage_path"])
        lib_mgr.update_status(video_id, "error", str(e))

def query_worker_process(session_id, query, global_config, server_url):
    """
    Worker for querying. Sets up UnifiedStorage.
    """
    import setproctitle
    setproctitle.setproctitle(f"videorag-query-{session_id}")
    
    try:
        sess_mgr = SessionManager(global_config["base_storage_path"])
        lib_mgr = LibraryManager(global_config["base_storage_path"])
        
        session = sess_mgr.get_session(session_id)
        if not session: raise ValueError("Session not found")
        
        # Collect VDB paths
        vdb_paths = []
        for vid in session["video_ids"]:
            # Library path: library/<vid>/vdb_chunks.json
            p = os.path.join(lib_mgr.library_dir, vid, "vdb_chunks.json")
            if os.path.exists(p):
                vdb_paths.append(p)
            else:
                 # Check if video_segment_feature vdb exists? 
                 # Unified storage usually just merges chunks (text) or all VDBs?
                 # My UnifiedStorage impl looked for `unified_vdb_{namespace}.json`
                 # and merged from source paths.
                 pass
        
        # We need to unify "chunks" (for text retrieval) and maybe "entities"?
        # For simplicity, UnifiedNanoVectorDBStorage implementation assumes it merges whatever is passed.
        # VideoRAG uses multiple VDBs: entities, chunks, video_segment_feature.
        # We need to merge ALL of them for "unified" experience?
        # Or just Text Chunks?
        # videorag_query checks entities_vdb, chunks_vdb, video_segment_feature_vdb.
        # So we need Unified version for ALL 3 types.
        
        session_dir = os.path.join(sess_mgr.sessions_dir, session_id)
        
        # We need to PREPARE the unified DBs before querying?
        # Or `UnifiedNanoVectorDBStorage` does it on init.
        # We should do it here.
        
        # 1. Chunks
        chunks_paths = [os.path.join(lib_mgr.library_dir, vid, "vdb_chunks.json") for vid in session["video_ids"]]
        
        # 2. Entities
        entities_paths = [os.path.join(lib_mgr.library_dir, vid, "vdb_entities.json") for vid in session["video_ids"]]
        
        # 3. Video Features
        features_paths = [os.path.join(lib_mgr.library_dir, vid, "vdb_video_segment_feature.json") for vid in session["video_ids"]]

        # Helper to init unified
        async def init_unified(paths, namespace):
             if not paths: return None
             u = UnifiedNanoVectorDBStorage(
                 namespace=namespace,
                 global_config={"working_dir": session_dir, "llm": {"embedding_batch_num": 32}},
                 embedding_func=openai_embedding, # Dummy, not used during merge really
                 source_vdb_paths=paths
             )
             await u.initialize_session_db()
             return u

        # We actually need to pass these configured classes to VideoRAG.
        # VideoRAG instantiation is complex.
        # We will dynamically override the storage classes used by this instance?
        # Or just pass the Unified class type?
        # VideoRAG accepts `vector_db_storage_cls`.
        # usage: `self.chunks_vdb = self.vector_db_storage_cls(...)`
        # But `UnifiedNanoVectorDBStorage` needs `source_vdb_paths`.
        # Standard VideoRAG doesn't know how to pass `source_vdb_paths`.
        
        # Approach:
        # We subclass VideoRAG or we just hack the instance.
        # Or we pre-initialize the Unified DBs, and then tell VideoRAG to use `NanoVectorDBStorage` 
        # but pointed to `unified_vdb_chunks.json`.
        
        # If we pre-initialize `unified_vdb_chunks.json` in `session_dir`,
        # And then Initialize VideoRAG with `working_dir=session_dir` and namespace="chunks",
        # `NanoVectorDBStorage` will look for `vdb_chunks.json`.
        # My UnifiedStorage created `unified_vdb_chunks.json`.
        # I should rename it to `vdb_chunks.json` inside the session dir!
        # Then standard VideoRAG will pick it up as if it's a local VDB.
        # THIS IS THE CLEANEST WAY! no code change in VideoRAG query logic.
        
        # So: Merge VDBs -> Save as `vdb_chunks.json` in session dir.
        # VideoRAG(working_dir=session_dir) -> reads `vdb_chunks.json`.
        
        # Implementation of Merge:
        from nano_vectordb import NanoVectorDB
        
        def merge_vdb(source_paths, target_path, dim):
            # Create/Reset target
            client = NanoVectorDB(dim, storage_file=target_path)
            # Load sources
            data_buffer = []
            
            for sp in source_paths:
                if os.path.exists(sp):
                    try:
                        # Use NanoVectorDB to load the file (handles all decoding logic)
                        source_client = NanoVectorDB(dim, storage_file=sp)
                        
                        # Access internal storage to get data and vectors
                        # Name mangling for private attribute __storage
                        storage = getattr(source_client, '_NanoVectorDB__storage', {})
                        
                        if isinstance(storage, dict) and "data" in storage and "matrix" in storage:
                            items = storage["data"]
                            matrix = storage["matrix"]
                            
                            # NanoVectorDB loads 'matrix' as a numpy array and 'data' as list
                            # We can trust that loaded lengths match or are handled by lib
                            valid_count = len(items)
                            
                            if len(matrix) >= valid_count:
                                for i, item in enumerate(items):
                                    new_item = item.copy() 
                                    new_item["__vector__"] = matrix[i]
                                    data_buffer.append(new_item)
                            else:
                                log_to_file(f"Mismatch in VDB {sp}: {valid_count} items vs {len(matrix)} vectors")
                                
                    except Exception as e:
                        log_to_file(f"Failed to merge VDB {sp}: {e}")

            if data_buffer:
                # Upsert handles everything
                client.upsert(data_buffer)
                
            client.save()
            return True
            
        # Merge Chunks (Dim 1536)
        merge_vdb(chunks_paths, os.path.join(session_dir, "vdb_chunks.json"), 1536)
        # Merge Entities (Dim 1536)
        merge_vdb(entities_paths, os.path.join(session_dir, "vdb_entities.json"), 1536)
        # Merge Video Features (Dim 1024)
        merge_vdb(features_paths, os.path.join(session_dir, "vdb_video_segment_feature.json"), 1024)
        
        # Also need to merge KVs? `text_chunks`, `video_segments`, `video_path`?
        # VideoRAG uses `JsonKVStorage`.
        # `text_chunks` (kv_store_text_chunks.json) - maps ID to content.
        # Yes, we need these. `NanoVectorDB` only stores vectors + meta fields by ID.
        # VideoRAG retrieves full content from KV store.
        
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

        # Merge Graph? (chunk_entity_relation)
        # It's a graphml file? Or storage? 
        # `NetworkXStorage` uses `graph_{namespace}.graphml`.
        # Merging graphml files is harder. 
        # But VideoRAG uses graph for `videorag_query` (GraphRAG).
        # Should we assume Graph is per video and we just union them?
        # NetworkX has `compose` or `union`.
        # If we skip graph merge, GraphRAG might not see cross-video connections (which represents new knowledge anyway).
        # We can try to load all graphs and merge them using networkx.
        
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

        # SETUP DONE. Now Query.
        
        imagebind_client = HTTPImageBindClient(server_url)
        videorag_llm_config = LLMConfig(
             # Same config as usual
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
            working_dir=session_dir, # Points to Unified Data
            ali_dashscope_api_key=global_config.get("ali_dashscope_api_key"),
            ali_dashscope_base_url=global_config.get("ali_dashscope_base_url"),
            caption_model=global_config.get("caption_model"),
            asr_model=global_config.get("asr_model"),
            openai_api_key=global_config.get("openai_api_key"),
            openai_base_url=global_config.get("openai_base_url"),
            imagebind_client=imagebind_client,
        )
        
        
        param = QueryParam(mode="videorag")
        param.wo_reference = True
        response = rag.query(query=query, param=param)
        
        # History Handling
        history_file = os.path.join(session_dir, "history.json")
        history = ensure_json_file(history_file, [])
        
        # NOTE: User message is already appended by endpoint.
        
        # Append AI Msg
        history.append({
            "role": "assistant", 
            "content": response, 
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        save_json_file(history_file, history)
        
        # Update Status
        status_file = os.path.join(session_dir, "status.json")
        res = {"query_status": {"status": "completed", "answer": response, "query": query}}
        save_json_file(status_file, res)
        
        # Update Session Last Active
        # We need to lock or be careful with session.json if multiple write?
        # SessionManager writes to single sessions.json.
        # Worker process should probably NOT write to sessions.json directly if concurrency is high.
        # But for MVP, we can read-modify-write.
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
        # Update status
        try:
            status_file = os.path.join(sess_mgr.sessions_dir, session_id, "status.json")
            save_json_file(status_file, {"query_status": {"status": "error", "message": str(e)}})
        except: pass


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
        "image_bind_model_path": os.getenv("IMAGE_BIND_MODEL_PATH"),
    }
    # Init ImageBind
    if config["image_bind_model_path"]:
        global_imagebind_manager.initialize(config["image_bind_model_path"])
        # We assume auto-load?
        # global_imagebind_manager.ensure_imagebind_loaded() 
    return True, "Config Loaded", config

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Load config on startup
    load_and_init_global_config()

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok"})
        
    @app.route("/api/initialize", methods=["POST"])
    def initialize():
        s, m, _ = load_and_init_global_config()
        return jsonify({"success": s, "message": m})

    # --- Library Endpoints ---
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
            "status": status.get("query_status")
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
             save_json_file(status_path, {"query_status": {"status": "processing"}})
             
        except Exception as e:
             log_to_file(f"Failed to sync save history/status: {e}")
        
        # 2. Spawn Worker for AI Response
        p = multiprocessing.Process(target=query_worker_process, args=(session_id, query, config, server_url))
        p.start()
        
        return jsonify({"success": True, "status": "processing"})
        
    @app.route("/api/sessions/<session_id>/status", methods=["GET"])
    def get_query_status(session_id):
        # ... existing implementation is fine, but maybe redundant if get_session_details exists
        pass # keeping old endpoint just in case, or remove if unused.
        sm = get_session_manager()
        p = os.path.join(sm.sessions_dir, session_id, "status.json")
        if os.path.exists(p):
             with open(p, 'r') as f:
                 return jsonify({"success": True, "status": json.load(f).get("query_status")})
        return jsonify({"success": False, "status": "unknown"})

    # --- ImageBind Internal ---
    @app.route("/api/imagebind/status", methods=["GET"])
    def ib_status():
        return jsonify({"success": True, "status": global_imagebind_manager.get_status()})
        
    @app.route("/api/imagebind/encode/video", methods=["POST"])
    def ib_encode_video():
        try:
             res = global_imagebind_manager.encode_video_segments(request.json["video_batch"])
             return jsonify({"success": True, "result": base64.b64encode(pickle.dumps(res)).decode()})
        except Exception as e:
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
