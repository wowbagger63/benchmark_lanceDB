import lancedb
import sqlite3
import numpy as np
import time
import shutil
import os

# Parameters
num_vectors = 1000
dim         = 1536

# Clean up
if os.path.exists("./lancedb_batch_bench"):
    shutil.rmtree("./lancedb_batch_bench")

# Prepare Data
print(f"Generating {num_vectors} vectors (dim={dim})...")
vectors    = np.random.random((num_vectors, dim)).astype(np.float32)
categories = np.random.choice(["legal", "other"], size=num_vectors)

# ─────────────────────────────────────────────
# 1. SQLite — Batch Insert
# ─────────────────────────────────────────────
print("\n--- SQLite Batch Insert ---")

# One row at a time
conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE vecs (id INTEGER PRIMARY KEY, category TEXT, vec BLOB)")
conn.commit()

print("SQLite: inserting one row at a time...")
start = time.perf_counter()
for i in range(num_vectors):
    conn.execute(
        "INSERT INTO vecs VALUES (?, ?, ?)",
        (i, categories[i], vectors[i].tobytes())
    )
conn.commit()
sqlite_one_by_one_ms = (time.perf_counter() - start) * 1000
print(f"  One-by-one insert:    {sqlite_one_by_one_ms:.2f} ms")

# executemany batch
conn2 = sqlite3.connect(":memory:")
conn2.execute("CREATE TABLE vecs (id INTEGER PRIMARY KEY, category TEXT, vec BLOB)")
conn2.commit()

print("SQLite: inserting with executemany (batch)...")
start = time.perf_counter()
conn2.executemany(
    "INSERT INTO vecs VALUES (?, ?, ?)",
    [(i, categories[i], vectors[i].tobytes()) for i in range(num_vectors)]
)
conn2.commit()
sqlite_batch_ms = (time.perf_counter() - start) * 1000
print(f"  executemany batch:    {sqlite_batch_ms:.2f} ms")

# With transaction
conn3 = sqlite3.connect(":memory:")
conn3.execute("CREATE TABLE vecs (id INTEGER PRIMARY KEY, category TEXT, vec BLOB)")
conn3.commit()

print("SQLite: inserting with explicit transaction...")
start = time.perf_counter()
with conn3:
    conn3.executemany(
        "INSERT INTO vecs VALUES (?, ?, ?)",
        [(i, categories[i], vectors[i].tobytes()) for i in range(num_vectors)]
    )
sqlite_transaction_ms = (time.perf_counter() - start) * 1000
print(f"  Explicit transaction: {sqlite_transaction_ms:.2f} ms")

conn.close()
conn2.close()
conn3.close()

# ─────────────────────────────────────────────
# 2. LanceDB — Batch Insert
# ─────────────────────────────────────────────
print("\n--- LanceDB Batch Insert ---")
db = lancedb.connect("./lancedb_batch_bench")

# Full batch — all 1000 at once
print("LanceDB: inserting all 1000 vectors at once...")
data  = [{"id": i, "vector": vectors[i], "category": categories[i]} for i in range(num_vectors)]
start = time.perf_counter()
table = db.create_table("vectors_full", data=data, mode="overwrite")
lancedb_full_batch_ms = (time.perf_counter() - start) * 1000
print(f"  Full batch (1000):    {lancedb_full_batch_ms:.2f} ms")

# Mini-batches of 100
print("LanceDB: inserting in mini-batches of 100...")
table2 = db.create_table(
    "vectors_mini",
    data=[{"id": 0, "vector": vectors[0], "category": categories[0]}],
    mode="overwrite"
)
start = time.perf_counter()
batch_size = 100
for batch_start in range(0, num_vectors, batch_size):
    batch = [
        {"id": i, "vector": vectors[i], "category": categories[i]}
        for i in range(batch_start, min(batch_start + batch_size, num_vectors))
    ]
    table2.add(batch)
lancedb_mini_batch_ms = (time.perf_counter() - start) * 1000
print(f"  Mini-batch (100x10):  {lancedb_mini_batch_ms:.2f} ms")

# Mini-batches of 10
print("LanceDB: inserting in mini-batches of 10...")
table3 = db.create_table(
    "vectors_tiny",
    data=[{"id": 0, "vector": vectors[0], "category": categories[0]}],
    mode="overwrite"
)
start = time.perf_counter()
batch_size = 10
for batch_start in range(0, num_vectors, batch_size):
    batch = [
        {"id": i, "vector": vectors[i], "category": categories[i]}
        for i in range(batch_start, min(batch_start + batch_size, num_vectors))
    ]
    table3.add(batch)
lancedb_tiny_batch_ms = (time.perf_counter() - start) * 1000
print(f"  Tiny-batch (10x100):  {lancedb_tiny_batch_ms:.2f} ms")

# One by one
print("LanceDB: inserting one vector at a time...")
table4 = db.create_table(
    "vectors_one",
    data=[{"id": 0, "vector": vectors[0], "category": categories[0]}],
    mode="overwrite"
)
start = time.perf_counter()
for i in range(1, num_vectors):
    table4.add([{"id": i, "vector": vectors[i], "category": categories[i]}])
lancedb_one_by_one_ms = (time.perf_counter() - start) * 1000
print(f"  One-by-one (1x1000):  {lancedb_one_by_one_ms:.2f} ms")

# ─────────────────────────────────────────────
# 3. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'BATCH INSERT BENCHMARK SUMMARY':^65}")
print(f"{'Vectors: 1,000  |  Dimensions: 1,536':^65}")
print("=" * 65)

print(f"\n{'── SQLite ──'}")
print(f"{'Method':<35} {'Time (ms)':>12} {'Per Vector':>14}")
print("-" * 65)
print(f"{'One-by-one insert':<35} {sqlite_one_by_one_ms:>11.2f}ms {sqlite_one_by_one_ms/num_vectors:>12.4f}ms")
print(f"{'executemany batch':<35} {sqlite_batch_ms:>11.2f}ms {sqlite_batch_ms/num_vectors:>12.4f}ms")
print(f"{'Explicit transaction':<35} {sqlite_transaction_ms:>11.2f}ms {sqlite_transaction_ms/num_vectors:>12.4f}ms")

print(f"\n{'── LanceDB ──'}")
print(f"{'Method':<35} {'Time (ms)':>12} {'Per Vector':>14}")
print("-" * 65)
print(f"{'Full batch (1000 at once)':<35} {lancedb_full_batch_ms:>11.2f}ms {lancedb_full_batch_ms/num_vectors:>12.4f}ms")
print(f"{'Mini-batch (100 per batch)':<35} {lancedb_mini_batch_ms:>11.2f}ms {lancedb_mini_batch_ms/num_vectors:>12.4f}ms")
print(f"{'Tiny-batch (10 per batch)':<35} {lancedb_tiny_batch_ms:>11.2f}ms {lancedb_tiny_batch_ms/num_vectors:>12.4f}ms")
print(f"{'One-by-one (1 per insert)':<35} {lancedb_one_by_one_ms:>11.2f}ms {lancedb_one_by_one_ms/num_vectors:>12.4f}ms")

print(f"\n{'── Fastest vs Fastest ──'}")
best_sqlite  = min(sqlite_one_by_one_ms, sqlite_batch_ms, sqlite_transaction_ms)
best_lancedb = min(lancedb_full_batch_ms, lancedb_mini_batch_ms, lancedb_tiny_batch_ms, lancedb_one_by_one_ms)
winner = "LanceDB" if best_lancedb < best_sqlite else "SQLite"
ratio  = max(best_sqlite, best_lancedb) / min(best_sqlite, best_lancedb)
print(f"  Best SQLite:  {best_sqlite:.2f}ms")
print(f"  Best LanceDB: {best_lancedb:.2f}ms")
print(f"  Winner: {winner} is {ratio:.1f}x faster")
print("=" * 65)