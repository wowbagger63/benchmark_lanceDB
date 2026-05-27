import lancedb
import numpy as np
import time

# 1. Initialization: Embedding and Setup
# In the lifecycle, the app uses a local ONNX runtime model.
def get_embedding(query_text):
    # This simulates the embedding generation (e.g., using ONNX/SentenceTransformers)
    # The result is a 1536-dimensional vector.
    return np.random.random(1536).astype(np.float32)

# Connect to the dataset
db = lancedb.connect("./lancedb_bench")
table = db.open_table("vectors")

def execute_query_lifecycle(query_text, category_filter, top_k=5):
    start_time = time.time()
    
    # 2. Embedding Generation (Total time includes this)
    query_vec = get_embedding(query_text)
    
    # 3. Query Construction and Execution
    # LanceDB performs the manifest read, filter evaluation (bitmap), 
    # and IVF-PQ partition search internally.
    results = (
        table.search(query_vec)
             .where(f"category = '{category_filter}'") # Metadata filter
             .limit(top_k)
             .to_list()
    )
    
    # 4. Result Materialization
    # The engine uses row-to-offset mapping for text retrieval.
    # The Arrow record batch is ready here for the LLM.
    total_elapsed = (time.time() - start_time) * 1000 # Convert to ms
    
    return results, total_elapsed

# Example Usage
query = "breach of contract from last month"
results, duration = execute_query_lifecycle(query, "legal")

print(f"Total Lifecycle Elapsed Time: {duration:.2f} ms")
print(f"Top {len(results)} documents retrieved for LLM processing.")