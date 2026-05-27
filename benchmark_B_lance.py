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

print("Creating IVF_PQ index...")
try:
    table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=256,
        num_sub_vectors=96,
    )
except TypeError:
    try:
        table.create_index(
            column="vector",
            index_type="IVF_PQ",
            metric="cosine",
            num_partitions=256,
            num_sub_vectors=96,
        )
    except TypeError:
        table.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=256,
            num_sub_vectors=96,
        )

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
# 3. Approximate Search — IVF_PQ (nprobes tuning)
# ─────────────────────────────────────────────
print("\n--- Approximate Search: IVF_PQ with nprobes tuning ---")
print(
    "IVF_PQ splits the vector space into partitions (IVF) and compresses\n"
    "each vector using Product Quantization (PQ). At query time only\n"
    "'nprobes' partitions are searched — higher nprobes = more accurate\n"
    "but slower; lower nprobes = faster but approximate.\n"
)

nprobes_list = [1, 5, 10, 20, 50, 100]
ivf_pq_results = []

for nprobes in nprobes_list:
    start = time.time()
    res   = (
        table.search(query_vec)
             .nprobes(nprobes)
             .limit(k)
             .to_list()
    )
    latency = time.time() - start
    ivf_pq_results.append((nprobes, latency, len(res)))
    print(f"  nprobes={nprobes:<4}  latency={latency:.4f}s  results={len(res)}")

# IVF_PQ WITH filter across nprobes
print("\nIVF_PQ WITH metadata filter (category = 'legal') across nprobes:")
ivf_pq_filtered_results = []

for nprobes in nprobes_list:
    start = time.time()
    res   = (
        table.search(query_vec)
             .nprobes(nprobes)
             .where("category = 'legal'")
             .limit(k)
             .to_list()
    )
    latency = time.time() - start
    ivf_pq_filtered_results.append((nprobes, latency, len(res)))
    print(f"  nprobes={nprobes:<4}  latency={latency:.4f}s  results={len(res)}")

# ─────────────────────────────────────────────
# 4. Recall Estimation — IVF_PQ vs brute force
# ─────────────────────────────────────────────
print("\n--- Recall Estimation: IVF_PQ vs Exact Search ---")
print("Computing exact top-k via brute force on a 10k sample...")

sample_size   = 10000
sample_vecs   = vectors[:sample_size]
sample_cats   = categories[:sample_size]

# Exact brute force on sample
start         = time.time()
dots          = sample_vecs @ query_vec
norms         = np.linalg.norm(sample_vecs, axis=1) * np.linalg.norm(query_vec)
cosine_sims   = dots / norms
exact_top_k   = set(np.argsort(cosine_sims)[-k:].tolist())
exact_time    = time.time() - start
print(f"Exact search latency (10k sample): {exact_time:.4f}s")

# IVF_PQ on same sample via LanceDB
sample_data  = [{"vector": sample_vecs[i], "category": sample_cats[i]} for i in range(sample_size)]
sample_table = db.create_table("sample_vectors", data=sample_data, mode="overwrite")

try:
    sample_table.create_index(
        metric="cosine",
        vector_column_name="vector",
        index_type="IVF_PQ",
        num_partitions=32,
        num_sub_vectors=96,
    )
except TypeError:
    sample_table.create_index(
        column="vector",
        index_type="IVF_PQ",
        metric="cosine",
        num_partitions=32,
        num_sub_vectors=96,
    )

recall_rows = []
for nprobes in [1, 5, 10, 20]:
    start  = time.time()
    res    = (
        sample_table.search(query_vec)
                    .nprobes(nprobes)
                    .limit(k)
                    .to_list()
    )
    latency    = time.time() - start
    # Recall = fraction of exact top-k found by approximate search
    approx_ids = set(range(len(res)))
    recall     = len(exact_top_k & approx_ids) / k
    recall_rows.append((nprobes, latency, recall))
    print(f"  nprobes={nprobes:<4}  latency={latency:.4f}s  recall@{k}={recall:.2%}")

# ─────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'BENCHMARK SUMMARY':^65}")
print("=" * 65)
print(f"{'Test':<36} {'Latency':>10} {'vs SQLite':>12}")
print("-" * 65)
print(f"{'SQLite   (filtered)':<36} {sqlite_filtered:>9.4f}s  {'baseline':>12}")
print(f"{'SQLite   (unfiltered)':<36} {sqlite_unfiltered:>9.4f}s  {'baseline':>12}")
print(f"{'LanceDB  IVF_PQ (filtered)':<36} {lancedb_filtered:>9.4f}s  {sqlite_filtered/lancedb_filtered:>9.1f}x faster")
print(f"{'LanceDB  IVF_PQ (unfiltered)':<36} {lancedb_unfiltered:>9.4f}s  {sqlite_unfiltered/lancedb_unfiltered:>9.1f}x faster")

print("\n--- IVF_PQ nprobes tradeoff (unfiltered) ---")
print(f"  {'nprobes':<10} {'latency':>10} {'results':>10}")
for nprobes, latency, count in ivf_pq_results:
    print(f"  {nprobes:<10} {latency:>9.4f}s {count:>10}")

print("\n--- IVF_PQ nprobes tradeoff (filtered) ---")
print(f"  {'nprobes':<10} {'latency':>10} {'results':>10}")
for nprobes, latency, count in ivf_pq_filtered_results:
    print(f"  {nprobes:<10} {latency:>9.4f}s {count:>10}")

print("\n--- Recall@10 vs nprobes (10k sample) ---")
print(f"  {'nprobes':<10} {'latency':>10} {'recall@10':>12}")
for nprobes, latency, recall in recall_rows:
    print(f"  {nprobes:<10} {latency:>9.4f}s {recall:>11.2%}")

print("=" * 65)
print(f"\nFilter impact — SQLite:  {sqlite_unfiltered/sqlite_filtered:.2f}x slower without filter")
print(f"Filter impact — LanceDB: {lancedb_unfiltered/lancedb_filtered:.2f}x slower without filter")