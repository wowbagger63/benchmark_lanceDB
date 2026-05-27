import lancedb
import sqlite3
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
import shutil

# ========================= PARAMETERS =========================
num_vectors = 1_000_000
num_threads = 4
dim = 1536

# ========================= DATA GENERATION =========================
print("Generating data...")
np.random.seed(42)  # for reproducibility
vectors = np.random.random((num_vectors, dim)).astype(np.float32)
categories = np.random.choice(["legal", "other"], size=num_vectors)

# Create 4 query vectors
query_vecs = [np.random.random(dim).astype(np.float32) for _ in range(num_threads)]

# ========================= SQLITE SETUP =========================
sqlite_path = "bench_concurrent.db"
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)

conn = sqlite3.connect(sqlite_path)
conn.execute("CREATE TABLE vecs (category TEXT, vec BLOB)")
conn.executemany(
    "INSERT INTO vecs VALUES (?, ?)",
    zip(categories, [v.tobytes() for v in vectors])
)
conn.commit()
conn.close()

# ========================= LANCEDB SETUP =========================
db_path = "./lancedb_bench"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

db = lancedb.connect(db_path)
data = [{"vector": vectors[i], "category": categories[i]} for i in range(num_vectors)]

table = db.create_table("vectors", data=data, mode="overwrite")

# FIXED: Correct way to create index in current LanceDB versions
print("Creating LanceDB index...")
table.create_index(
    metric="cosine",
    vector_column_name="vector",
    num_partitions=256,
    num_sub_vectors=96
)

print("Setup complete. Running benchmark...\n")

# ========================= BENCHMARK FUNCTIONS =========================
def sqlite_read(qvec):
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT vec FROM vecs WHERE category = 'legal'")
    rows = cur.fetchall()
    for row in rows:
        sv = np.frombuffer(row[0], dtype=np.float32)
        _ = np.dot(sv, qvec)  # simulate work
    conn.close()


def lancedb_read(qvec):
    results = (
        table.search(qvec)
        .where("category = 'legal'")
        .limit(10)
        .to_list()
    )
    return results


# ========================= RUN BENCHMARK =========================
def run_benchmark(name, func):
    print(f"Running {num_threads} concurrent {name} reads...")
    start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(func, query_vecs)
    
    duration_ms = (time.perf_counter() - start) * 1000
    print(f"{name} total wall time: {duration_ms:.2f} ms\n")
    return duration_ms


sqlite_time = run_benchmark("SQLite", sqlite_read)
lancedb_time = run_benchmark("LanceDB", lancedb_read)

print("="*60)
print("FINAL RESULTS")
print(f"SQLite  (4 concurrent reads): {sqlite_time:8.2f} ms")
print(f"LanceDB (4 concurrent reads): {lancedb_time:8.2f} ms")
print("="*60)