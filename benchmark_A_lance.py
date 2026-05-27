import lancedb
import sqlite3
import numpy as np
import time
import shutil
import os

# Parameters
num_vectors  = 1000000
dim          = 1536
k            = 10
filter_ratio = 0.15

# Clean up
if os.path.exists("./lancedb_bench"):
    shutil.rmtree("./lancedb_bench")

# Prepare Data
print(f"Generating {num_vectors} vectors...")
vectors    = np.random.random((num_vectors, dim)).astype(np.float32)
categories = np.random.choice(["legal", "other"], size=num_vectors, p=[filter_ratio, 1 - filter_ratio])
query_vec  = np.random.random(dim).astype(np.float32)

# ─────────────────────────────────────────────
# 1. SQLite Benchmark — WITH filter
# ─────────────────────────────────────────────
print("\n--- SQLite ---")
conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE vecs (category TEXT, vec BLOB)")
conn.executemany("INSERT INTO vecs VALUES (?, ?)", zip(categories, [v.tobytes() for v in vectors]))
conn.commit()

print("Running SQLite WITH filter...")
start  = time.time()
cursor = conn.cursor()
cursor.execute("SELECT vec FROM vecs WHERE category = 'legal'")
rows = cursor.fetchall()
for row in rows:
    stored_vec = np.frombuffer(row[0], dtype=np.float32)
    dist = np.dot(stored_vec, query_vec) / (np.linalg.norm(stored_vec) * np.linalg.norm(query_vec))
sqlite_filtered = time.time() - start
print(f"SQLite latency (filtered):   {sqlite_filtered:.4f}s  |  Rows scanned: {len(rows)}")

# SQLite — WITHOUT filter
print("Running SQLite WITHOUT filter...")
start  = time.time()
cursor = conn.cursor()
cursor.execute("SELECT vec FROM vecs")
rows = cursor.fetchall()
for row in rows:
    stored_vec = np.frombuffer(row[0], dtype=np.float32)
    dist = np.dot(stored_vec, query_vec) / (np.linalg.norm(stored_vec) * np.linalg.norm(query_vec))
sqlite_unfiltered = time.time() - start
print(f"SQLite latency (unfiltered): {sqlite_unfiltered:.4f}s  |  Rows scanned: {len(rows)}")

conn.close()

# ─────────────────────────────────────────────
# 2. LanceDB Benchmark — WITH and WITHOUT filter
# ─────────────────────────────────────────────
print("\n--- LanceDB ---")
print("Creating LanceDB table...")
db    = lancedb.connect("./lancedb_bench")
data  = [{"vector": vectors[i], "category": categories[i]} for i in range(num_vectors)]
table = db.create_table("vectors", data=data, mode="overwrite")

print("Creating index...")
try:
    # Modern LanceDB API (0.6+)
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=256,
        num_sub_vectors=96,
    )
except TypeError:
    try:
        # Mid-range versions
        table.create_index(
            column="vector",
            index_type="IVF_PQ",
            metric="cosine",
            num_partitions=256,
            num_sub_vectors=96,
        )
    except TypeError:
        # Older versions
        table.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=256,
            num_sub_vectors=96,
        )

# LanceDB — WITH filter
print("Running LanceDB WITH filter...")
start   = time.time()
results = (
    table.search(query_vec)
         .where("category = 'legal'")
         .limit(k)
         .to_list()
)
lancedb_filtered = time.time() - start
print(f"LanceDB latency (filtered):   {lancedb_filtered:.4f}s  |  Results: {len(results)}")

# LanceDB — WITHOUT filter
print("Running LanceDB WITHOUT filter...")
start   = time.time()
results = (
    table.search(query_vec)
         .limit(k)
         .to_list()
)
lancedb_unfiltered = time.time() - start
print(f"LanceDB latency (unfiltered): {lancedb_unfiltered:.4f}s  |  Results: {len(results)}")

# ─────────────────────────────────────────────
# 3. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'BENCHMARK SUMMARY':^55}")
print("=" * 55)
print(f"{'Test':<30} {'Latency':>10} {'Speedup':>12}")
print("-" * 55)
print(f"{'SQLite   (filtered)':<30} {sqlite_filtered:>9.4f}s")
print(f"{'SQLite   (unfiltered)':<30} {sqlite_unfiltered:>9.4f}s")
print(f"{'LanceDB  (filtered)':<30} {lancedb_filtered:>9.4f}s  {sqlite_filtered/lancedb_filtered:>8.1f}x faster")
print(f"{'LanceDB  (unfiltered)':<30} {lancedb_unfiltered:>9.4f}s  {sqlite_unfiltered/lancedb_unfiltered:>8.1f}x faster")
print("=" * 55)
print(f"\nFilter impact — SQLite:  {sqlite_unfiltered/sqlite_filtered:.2f}x slower without filter")
print(f"Filter impact — LanceDB: {lancedb_unfiltered/lancedb_filtered:.2f}x slower without filter")