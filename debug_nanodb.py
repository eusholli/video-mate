
import os
import json
import numpy as np
from nano_vectordb import NanoVectorDB

# Path to the library VDB file
vdb_path = "backend/ds/storage/library/c0668c553e33532427f4904ec766aad9/vdb_chunks.json"
target_path = "temp_merged_test.json"

if os.path.exists(target_path): os.remove(target_path)

def merge_vdb_safe(source_paths, target_path, dim):
    target_client = NanoVectorDB(dim, storage_file=target_path)
    data_buffer = []

    for sp in source_paths:
        if os.path.exists(sp):
            try:
                print(f"Loading source VDB via NanoVectorDB: {sp}")
                # Load source
                source_client = NanoVectorDB(dim, storage_file=sp)
                
                # Check loaded data
                # NanoVectorDB stores everything in memory on init
                # We need to access the data. 
                # Inspecting common attributes:
                
                # Inspect storage
                storage = getattr(source_client, '_NanoVectorDB__storage', None)
                if storage:
                    print(f"Storage type: {type(storage)}")
                    print(f"Storage keys: {storage.keys() if isinstance(storage, dict) else dir(storage)}")
                    if isinstance(storage, dict):
                        # Assuming storage dict holds data
                        if "data" in storage:
                            print(f"Found 'data' in storage. Len: {len(storage['data'])}")
                            if storage['data']:
                                print(f"First item in 'data': {storage['data'][0]}")
                        if "matrix" in storage:
                             print(f"Found 'matrix' in storage. Len: {len(storage['matrix'])}")
                             print(f"Matrix type: {type(storage['matrix'])}")
                             if len(storage['matrix']) > 0:
                                 vec = storage['matrix'][0]
                                 print(f"Vector 0 type: {type(vec)}")
                                 if isinstance(vec, np.ndarray):
                                     print(f"Vector 0 shape: {vec.shape}")
                pass

            except Exception as e:
                print(f"Error loading source {sp}: {e}")

    return False

merge_vdb_safe([vdb_path], target_path, 1536)
