import lancedb
import sqlite3
import numpy as np
import time
import shutil
import os

# Parameters
num_vectors = 1000
dim         = 1536

# Paths
sqlite_path = "./sqlite_bench.db"
lancedb_path = "./lancedb_batch_bench"

# Clean up
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)
if os.path.exists(lancedb_path):
    shutil.rmtree(lancedb_path)

# Prepare Data
print(f"Generating {num_vectors} vectors (dim={dim})...")
vectors    = np.random.random((num_vectors, dim)).astype(np.float32)
categories = np.random.choice(["legal", "other"], size=num_vectors)
blobs      = [v.tobytes() for v in vectors]
lance_data = [{"id": i, "vector": vectors[i], "category": categories[i]} for i in range(num_vectors)]

# ─────────────────────────────────────────────
# Warmup — excluded from timing
# ─────────────────────────────────────────────
print("Warming up...")

w_sqlite = sqlite3.connect(":memory:")
w_sqlite.execute("CREATE TABLE w (id INTEGER, vec BLOB)")
w_sqlite.execute("INSERT INTO w VALUES (0, ?)", (blobs[0],))
w_sqlite.commit()
w_sqlite.close()

db = lancedb.connect(lancedb_path)
db.create_table("warmup", data=[lance_data[0]], mode="overwrite")

# ─────────────────────────────────────────────
# 1. SQLite — Batch Insert (on disk)
# ─────────────────────────────────────────────
print("\n--- SQLite Batch Insert (on-disk) ---")

# Run 3 times and take best
sqlite_times = []
for run in range(3):
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE vecs (id INTEGER PRIMARY KEY, category TEXT, vec BLOB)")
    conn.commit()

    start = time.perf_counter()
    conn.executemany(
        "INSERT INTO vecs VALUES (?, ?, ?)",
        [(i, categories[i], blobs[i]) for i in range(num_vectors)]
    )
    conn.commit()
    elapsed = (time.perf_counter() - start) * 1000
    sqlite_times.append(elapsed)
    conn.close()
    print(f"  Run {run+1}: {elapsed:.2f} ms")

sqlite_best_ms = min(sqlite_times)
sqlite_avg_ms  = np.mean(sqlite_times)
print(f"  Best: {sqlite_best_ms:.2f} ms  |  Avg: {sqlite_avg_ms:.2f} ms")

# ─────────────────────────────────────────────
# 2. LanceDB — Batch Insert (on disk)
# ─────────────────────────────────────────────
print("\n--- LanceDB Batch Insert (on-disk) ---")

# Pre-create table to exclude schema setup — same as SQLite CREATE TABLE above
table = db.create_table("vectors_bench", data=[lance_data[0]], mode="overwrite")

# Run 3 times and take best
lancedb_times = []
for run in range(3):
    # Reset table
    table = db.create_table("vectors_bench", data=[lance_data[0]], mode="overwrite")

    start   = time.perf_counter()
    table.add(lance_data[1:])   # batch insert remaining 999 vectors
    elapsed = (time.perf_counter() - start) * 1000
    lancedb_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.2f} ms")

lancedb_best_ms = min(lancedb_times)
lancedb_avg_ms  = np.mean(lancedb_times)
print(f"  Best: {lancedb_best_ms:.2f} ms  |  Avg: {lancedb_avg_ms:.2f} ms")

# ─────────────────────────────────────────────
# 3. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'BATCH INSERT BENCHMARK SUMMARY':^60}")
print(f"{'1,000 vectors  |  1,536 dimensions  |  3 runs each':^60}")
print("=" * 60)
print(f"\n{'Database':<20} {'Best (ms)':>12} {'Avg (ms)':>12} {'Per Vec (ms)':>14}")
print("-" * 60)
print(f"{'SQLite  (on-disk)':<20} {sqlite_best_ms:>11.2f}ms {sqlite_avg_ms:>11.2f}ms {sqlite_best_ms/num_vectors:>12.4f}ms")
print(f"{'LanceDB (on-disk)':<20} {lancedb_best_ms:>11.2f}ms {lancedb_avg_ms:>11.2f}ms {lancedb_best_ms/num_vectors:>12.4f}ms")

print(f"\n{'── Winner ──'}")
if lancedb_best_ms < sqlite_best_ms:
    ratio = sqlite_best_ms / lancedb_best_ms
    print(f"  ✓ LanceDB is {ratio:.1f}x faster than SQLite for batch insert")
else:
    ratio = lancedb_best_ms / sqlite_best_ms
    print(f"  ✓ SQLite is {ratio:.1f}x faster than LanceDB for batch insert")
    print(f"  Note: LanceDB stores columnar vector data with richer metadata.")
    print(f"        For pure insert speed at small scale SQLite can win.")
    print(f"        LanceDB advantage appears at scale with ANN search.")
print("=" * 60)

# Cleanup
if os.path.exists(sqlite_path):
    os.remove(sqlite_path)
if os.path.exists(lancedb_path):
    shutil.rmtree(lancedb_path)