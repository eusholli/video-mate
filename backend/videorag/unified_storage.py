from dataclasses import dataclass
import os
import shutil
from typing import List
from ._storage.vdb_nanovectordb import NanoVectorDBStorage
from ._utils import logger
from nano_vectordb import NanoVectorDB

@dataclass
class UnifiedNanoVectorDBStorage(NanoVectorDBStorage):
    """
    Unified storage that aggregates multiple video VDBs into a single session VDB.
    """
    source_vdb_paths: List[str] = None
    
    def __post_init__(self):
        # We override __post_init__ to control initialization
        pass

    async def initialize_session_db(self):
        """
        Explicit initialization: 
        1. Create a temporary VDB for this session.
        2. Load data from source_vdb_paths.
        3. Upsert into temp VDB.
        """
        if not self.source_vdb_paths:
            logger.warning("No source VDBs provided for Unified Storage")
            self.source_vdb_paths = []

        # Use the namespace (which should be unique per session) to define the specific temp file
        # namespace is typically passed as "chunks" or "entities"
        # self.global_config["working_dir"] is the session directory
        
        self.session_vdb_path = os.path.join(
            self.global_config["working_dir"], 
            f"unified_vdb_{self.namespace}.json"
        )
        
        # Always start fresh for a session load to ensure correctness
        if os.path.exists(self.session_vdb_path):
            os.remove(self.session_vdb_path)
            
        logger.info(f"Initializing Unified VDB at {self.session_vdb_path}")
        
        # Initialize the actual NanoVectorDB client
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, 
            storage_file=self.session_vdb_path
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", 0.2
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        
        # Load and Merge Data
        for source_path in self.source_vdb_paths:
            if not os.path.exists(source_path):
                logger.warning(f"Source VDB not found: {source_path}")
                continue
                
            try:
                # We load the source VDB using NanoVectorDB directly to read data
                source_client = NanoVectorDB(
                    self.embedding_func.embedding_dim, 
                    storage_file=source_path
                )
                
                # Get all data from source
                # NanoVectorDB doesn't have a simple "dump all" public method easily accessible in some versions,
                # but typically we can access storage directly or query all?
                # Looking at NanoVectorDB source or usage:
                # it stores data in self._client.datas usually?
                # We can just copy the data structure if it's JSON.
                # Actually, NanoVectorDB loads everything into memory.
                
                # If NanoVectorDB has a `get` method or we can read the JSON file directly.
                # Let's read the JSON file directly since we know the format.
                import json
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                    
                # source_data is list of dicts: [{"__id__": ..., "__vector__": ...}, ...]
                if source_data:
                    logger.info(f"Merging {len(source_data)} entries from {source_path}")
                    # Prepare for upsert
                    # NanoVectorDB.upsert expects list of dicts with __id__ and __vector__
                    # valid.
                    self._client.upsert(datas=source_data)
                    
            except Exception as e:
                logger.error(f"Failed to merge VDB {source_path}: {e}")
        
        self._client.save()
        logger.info(f"Unified VDB initialized with combined data.")

    # We need to make sure upsert/query works.
    # post_init in parent class sets up self._client. We disabled it.
    # So we MUST call initialize_session_db explicitly or in __post_init__ if we have params.
    # But parent dataclass structure makes it hard to pass extra params in __init__ cleanly without override.
    # So we used a field `source_vdb_paths`.
    
    # We call initialize_session_db manually after creation?
    # Or we call it in __post_init__ if safe.
    
    # Actually, we need to call it in __post_init__ but `initialize_session_db` is async?
    # upsert is async in parent class? No, `self._client.upsert` is sync, bu wrapper is async.
    # `initialize_session_db` can be synchronous because `upsert` and `json.load` are fast enough or we don't care (blocking on init).
    # NanoVectorDB operations are sync.
    
    # Let's change initialize_session_db to be sync and call it in __post_init__.
