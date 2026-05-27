import lancedb
import sqlite3
import numpy as np
import time
import shutil
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameters
num_vectors  = 1000000
dim          = 1536
k            = 10
filter_ratio = 0.15
num_threads  = 4

# Clean up
if os.path.exists("./lancedb_bench"):
    shutil.rmtree("./lancedb_bench")

# Prepare Data
print(f"Generating {num_vectors} vectors...")
vectors    = np.random.random((num_vectors, dim)).astype(np.float32)
categories = np.random.choice(["legal", "other"], size=num_vectors, p=[filter_ratio, 1 - filter_ratio])
query_vec  = np.random.random(dim).astype(np.float32)

# ─────────────────────────────────────────────
# 1. SQLite Benchmark — WITH and WITHOUT filter
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
# 2. LanceDB — WITH and WITHOUT filter
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
# 3. Approximate Search — IVF_PQ nprobes tuning
# ─────────────────────────────────────────────
print("\n--- Approximate Search: IVF_PQ with nprobes tuning ---")
print(
    "IVF_PQ splits the vector space into partitions (IVF) and compresses\n"
    "each vector using Product Quantization (PQ). At query time only\n"
    "'nprobes' partitions are searched — higher nprobes = more accurate\n"
    "but slower; lower nprobes = faster but approximate.\n"
)

nprobes_list         = [1, 5, 10, 20, 50, 100]
ivf_pq_results       = []
ivf_pq_filtered_results = []

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

print("\nIVF_PQ WITH metadata filter (category = 'legal') across nprobes:")
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

sample_size = 10000
sample_vecs = vectors[:sample_size]
sample_cats = categories[:sample_size]

start       = time.time()
dots        = sample_vecs @ query_vec
norms       = np.linalg.norm(sample_vecs, axis=1) * np.linalg.norm(query_vec)
cosine_sims = dots / norms
exact_top_k = set(np.argsort(cosine_sims)[-k:].tolist())
exact_time  = time.time() - start
print(f"Exact search latency (10k sample): {exact_time:.4f}s")

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
    start      = time.time()
    res        = (
        sample_table.search(query_vec)
                    .nprobes(nprobes)
                    .limit(k)
                    .to_list()
    )
    latency    = time.time() - start
    approx_ids = set(range(len(res)))
    recall     = len(exact_top_k & approx_ids) / k
    recall_rows.append((nprobes, latency, recall))
    print(f"  nprobes={nprobes:<4}  latency={latency:.4f}s  recall@{k}={recall:.2%}")

# ─────────────────────────────────────────────
# 5. Concurrent Read Load — 4 Threads
# ─────────────────────────────────────────────
print(f"\n--- Concurrent Read Load ({num_threads} threads) ---")
print(
    f"Simulates {num_threads} simultaneous readers each firing queries\n"
    "against SQLite and LanceDB to measure throughput and latency\n"
    "under concurrent load.\n"
)

# Generate unique query vectors per thread
thread_queries = [np.random.random(dim).astype(np.float32) for _ in range(num_threads)]

# ── SQLite concurrent — WITH filter ──
print(f"SQLite concurrent ({num_threads} threads) WITH filter...")

def sqlite_query_filtered(qvec):
    # Each thread needs its own connection — SQLite is not thread-safe with shared connections
    thread_conn = sqlite3.connect(":memory:")
    thread_conn.execute("CREATE TABLE vecs (category TEXT, vec BLOB)")
    thread_conn.executemany(
        "INSERT INTO vecs VALUES (?, ?)",
        zip(categories, [v.tobytes() for v in vectors])
    )
    thread_conn.commit()
    start  = time.time()
    cur    = thread_conn.cursor()
    cur.execute("SELECT vec FROM vecs WHERE category = 'legal'")
    rows   = cur.fetchall()
    for row in rows:
        sv   = np.frombuffer(row[0], dtype=np.float32)
        dist = np.dot(sv, qvec) / (np.linalg.norm(sv) * np.linalg.norm(qvec))
    elapsed = time.time() - start
    thread_conn.close()
    return elapsed

wall_start = time.time()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures  = [executor.submit(sqlite_query_filtered, q) for q in thread_queries]
    s_thread_times_filtered = [f.result() for f in as_completed(futures)]
sqlite_concurrent_filtered_wall = time.time() - wall_start
print(f"  Wall time:          {sqlite_concurrent_filtered_wall:.4f}s")
print(f"  Per-thread latency: min={min(s_thread_times_filtered):.4f}s  max={max(s_thread_times_filtered):.4f}s  avg={np.mean(s_thread_times_filtered):.4f}s")

# ── SQLite concurrent — WITHOUT filter ──
print(f"SQLite concurrent ({num_threads} threads) WITHOUT filter...")

def sqlite_query_unfiltered(qvec):
    thread_conn = sqlite3.connect(":memory:")
    thread_conn.execute("CREATE TABLE vecs (category TEXT, vec BLOB)")
    thread_conn.executemany(
        "INSERT INTO vecs VALUES (?, ?)",
        zip(categories, [v.tobytes() for v in vectors])
    )
    thread_conn.commit()
    start  = time.time()
    cur    = thread_conn.cursor()
    cur.execute("SELECT vec FROM vecs")
    rows   = cur.fetchall()
    for row in rows:
        sv   = np.frombuffer(row[0], dtype=np.float32)
        dist = np.dot(sv, qvec) / (np.linalg.norm(sv) * np.linalg.norm(qvec))
    elapsed = time.time() - start
    thread_conn.close()
    return elapsed

wall_start = time.time()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures  = [executor.submit(sqlite_query_unfiltered, q) for q in thread_queries]
    s_thread_times_unfiltered = [f.result() for f in as_completed(futures)]
sqlite_concurrent_unfiltered_wall = time.time() - wall_start
print(f"  Wall time:          {sqlite_concurrent_unfiltered_wall:.4f}s")
print(f"  Per-thread latency: min={min(s_thread_times_unfiltered):.4f}s  max={max(s_thread_times_unfiltered):.4f}s  avg={np.mean(s_thread_times_unfiltered):.4f}s")

# ── LanceDB concurrent — WITH filter ──
print(f"LanceDB concurrent ({num_threads} threads) WITH filter...")

def lancedb_query_filtered(qvec):
    start   = time.time()
    res     = (
        table.search(qvec)
             .where("category = 'legal'")
             .limit(k)
             .to_list()
    )
    return time.time() - start

wall_start = time.time()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures  = [executor.submit(lancedb_query_filtered, q) for q in thread_queries]
    l_thread_times_filtered = [f.result() for f in as_completed(futures)]
lancedb_concurrent_filtered_wall = time.time() - wall_start
print(f"  Wall time:          {lancedb_concurrent_filtered_wall:.4f}s")
print(f"  Per-thread latency: min={min(l_thread_times_filtered):.4f}s  max={max(l_thread_times_filtered):.4f}s  avg={np.mean(l_thread_times_filtered):.4f}s")

# ── LanceDB concurrent — WITHOUT filter ──
print(f"LanceDB concurrent ({num_threads} threads) WITHOUT filter...")

def lancedb_query_unfiltered(qvec):
    start   = time.time()
    res     = (
        table.search(qvec)
             .limit(k)
             .to_list()
    )
    return time.time() - start

wall_start = time.time()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures  = [executor.submit(lancedb_query_unfiltered, q) for q in thread_queries]
    l_thread_times_unfiltered = [f.result() for f in as_completed(futures)]
lancedb_concurrent_unfiltered_wall = time.time() - wall_start
print(f"  Wall time:          {lancedb_concurrent_unfiltered_wall:.4f}s")
print(f"  Per-thread latency: min={min(l_thread_times_unfiltered):.4f}s  max={max(l_thread_times_unfiltered):.4f}s  avg={np.mean(l_thread_times_unfiltered):.4f}s")

# ── LanceDB concurrent IVF_PQ with nprobes=10 ──
print(f"LanceDB concurrent ({num_threads} threads) IVF_PQ nprobes=10 WITHOUT filter...")

def lancedb_query_ivfpq(qvec):
    start = time.time()
    res   = (
        table.search(qvec)
             .nprobes(10)
             .limit(k)
             .to_list()
    )
    return time.time() - start

wall_start = time.time()
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures  = [executor.submit(lancedb_query_ivfpq, q) for q in thread_queries]
    l_thread_times_ivfpq = [f.result() for f in as_completed(futures)]
lancedb_concurrent_ivfpq_wall = time.time() - wall_start
print(f"  Wall time:          {lancedb_concurrent_ivfpq_wall:.4f}s")
print(f"  Per-thread latency: min={min(l_thread_times_ivfpq):.4f}s  max={max(l_thread_times_ivfpq):.4f}s  avg={np.mean(l_thread_times_ivfpq):.4f}s")

# ─────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'BENCHMARK SUMMARY':^70}")
print("=" * 70)

print(f"\n{'── Single-threaded ──'}")
print(f"{'Test':<40} {'Latency':>10} {'vs SQLite':>14}")
print("-" * 70)
print(f"{'SQLite   (filtered)':<40} {sqlite_filtered:>9.4f}s  {'baseline':>14}")
print(f"{'SQLite   (unfiltered)':<40} {sqlite_unfiltered:>9.4f}s  {'baseline':>14}")
print(f"{'LanceDB  IVF_PQ (filtered)':<40} {lancedb_filtered:>9.4f}s  {sqlite_filtered/lancedb_filtered:>11.1f}x faster")
print(f"{'LanceDB  IVF_PQ (unfiltered)':<40} {lancedb_unfiltered:>9.4f}s  {sqlite_unfiltered/lancedb_unfiltered:>11.1f}x faster")

print(f"\n{'── IVF_PQ nprobes tradeoff (unfiltered) ──'}")
print(f"  {'nprobes':<10} {'latency':>10} {'results':>10}")
for nprobes, latency, count in ivf_pq_results:
    print(f"  {nprobes:<10} {latency:>9.4f}s {count:>10}")

print(f"\n{'── IVF_PQ nprobes tradeoff (filtered) ──'}")
print(f"  {'nprobes':<10} {'latency':>10} {'results':>10}")
for nprobes, latency, count in ivf_pq_filtered_results:
    print(f"  {nprobes:<10} {latency:>9.4f}s {count:>10}")

print(f"\n{'── Recall@10 vs nprobes (10k sample) ──'}")
print(f"  {'nprobes':<10} {'latency':>10} {'recall@10':>12}")
for nprobes, latency, recall in recall_rows:
    print(f"  {nprobes:<10} {latency:>9.4f}s {recall:>11.2%}")

print(f"\n{'── Concurrent Read Load (4 threads) — Wall Time ──'}")
print(f"{'Test':<45} {'Wall Time':>10} {'vs SQLite':>12}")
print("-" * 70)
print(f"{'SQLite   concurrent (filtered)':<45} {sqlite_concurrent_filtered_wall:>9.4f}s  {'baseline':>12}")
print(f"{'SQLite   concurrent (unfiltered)':<45} {sqlite_concurrent_unfiltered_wall:>9.4f}s  {'baseline':>12}")
print(f"{'LanceDB  concurrent (filtered)':<45} {lancedb_concurrent_filtered_wall:>9.4f}s  {sqlite_concurrent_filtered_wall/lancedb_concurrent_filtered_wall:>9.1f}x faster")
print(f"{'LanceDB  concurrent (unfiltered)':<45} {lancedb_concurrent_unfiltered_wall:>9.4f}s  {sqlite_concurrent_unfiltered_wall/lancedb_concurrent_unfiltered_wall:>9.1f}x faster")
print(f"{'LanceDB  concurrent IVF_PQ nprobes=10':<45} {lancedb_concurrent_ivfpq_wall:>9.4f}s  {sqlite_concurrent_unfiltered_wall/lancedb_concurrent_ivfpq_wall:>9.1f}x faster")

print(f"\n{'── Concurrent Per-Thread Average Latency ──'}")
print(f"{'Test':<45} {'Avg':>8} {'Min':>8} {'Max':>8}")
print("-" * 70)
print(f"{'SQLite   concurrent (filtered)':<45} {np.mean(s_thread_times_filtered):>7.4f}s {min(s_thread_times_filtered):>7.4f}s {max(s_thread_times_filtered):>7.4f}s")
print(f"{'SQLite   concurrent (unfiltered)':<45} {np.mean(s_thread_times_unfiltered):>7.4f}s {min(s_thread_times_unfiltered):>7.4f}s {max(s_thread_times_unfiltered):>7.4f}s")
print(f"{'LanceDB  concurrent (filtered)':<45} {np.mean(l_thread_times_filtered):>7.4f}s {min(l_thread_times_filtered):>7.4f}s {max(l_thread_times_filtered):>7.4f}s")
print(f"{'LanceDB  concurrent (unfiltered)':<45} {np.mean(l_thread_times_unfiltered):>7.4f}s {min(l_thread_times_unfiltered):>7.4f}s {max(l_thread_times_unfiltered):>7.4f}s")
print(f"{'LanceDB  concurrent IVF_PQ nprobes=10':<45} {np.mean(l_thread_times_ivfpq):>7.4f}s {min(l_thread_times_ivfpq):>7.4f}s {max(l_thread_times_ivfpq):>7.4f}s")

print("=" * 70)
print(f"\nFilter impact — SQLite  single-thread: {sqlite_unfiltered/sqlite_filtered:.2f}x slower without filter")
print(f"Filter impact — LanceDB single-thread: {lancedb_unfiltered/lancedb_filtered:.2f}x slower without filter")
print(f"Filter impact — SQLite  concurrent:    {sqlite_concurrent_unfiltered_wall/sqlite_concurrent_filtered_wall:.2f}x slower without filter")
print(f"Filter impact — LanceDB concurrent:    {lancedb_concurrent_unfiltered_wall/lancedb_concurrent_filtered_wall:.2f}x slower without filter")