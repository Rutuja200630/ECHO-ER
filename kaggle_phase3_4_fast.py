"""
ECHO-ER Phase 3 & 4: FIXED Multi-View Sparse Retrieval
=======================================================
ROOT CAUSE OF HANG: ProcessPoolExecutor returning 16M+ numpy arrays
through IPC pipes causes deadlock.

FIX:
- Single-threaded BM25 querying (no IPC overhead)
- Write results chunk-by-chunk directly to disk (no big arrays in RAM)
- Progress print every 50k queries so you know it's alive
- Index built once & cached as .pkl, reused on restart
- Per-view checkpoints (skip already done views)
"""
import os
import sys
import time
import gc
import pickle
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# CONFIG
# ============================================================
TOP_K        = 30
BM25_K1      = 1.5
BM25_B       = 0.75
MAX_VOCAB    = 300_000
MIN_DF       = 2
CHUNK_WRITE  = 50_000      # write results every N queries

CACHE_DIR     = "/kaggle/working/cache"
OUT_DIR       = "/kaggle/working/retrieval"
PROCESSED_DIR = "/kaggle/working/data/processed/train"

VIEWS = {
    "name":    ["name_norm"],
    "address": ["address_norm", "postal_code"],
    "full":    ["name_norm", "address_norm", "country_norm", "postal_code"],
}

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# TOKENIZER
# ============================================================
_RE_TOKEN = re.compile(r'\b\w+\b')

def tokenize(text: str) -> List[str]:
    if not text or (isinstance(text, float) and math.isnan(text)):
        return []
    return _RE_TOKEN.findall(str(text).lower())

def build_text(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        v = row.get(c, "")
        if v and v != "nan":
            parts.append(str(v))
    return " ".join(parts)

# ============================================================
# BM25 INVERTED INDEX
# ============================================================
class BM25Index:
    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_ids = None
        self.doc_len = None
        self.avgdl = 0.0
        self.N = 0
        self.token2id: Dict[str, int] = {}
        self.postings: List[Tuple[np.ndarray, np.ndarray]] = []
        self.idf: np.ndarray = None

    def build(self, texts: List[str], doc_ids: np.ndarray):
        print("  [BM25] Tokenizing corpus...")
        t0 = time.time()
        self.N = len(texts)
        self.doc_ids = doc_ids

        raw_index: Dict[str, List] = defaultdict(list)
        doc_lengths = np.zeros(self.N, dtype=np.int32)

        for pos, text in enumerate(texts):
            tokens = tokenize(text)
            doc_lengths[pos] = len(tokens)
            if not tokens: continue
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                raw_index[t].append((pos, tf))

            if pos > 0 and pos % 1_000_000 == 0:
                print(f"    Tokenized {pos:,}/{self.N:,} docs...")

        self.doc_len = doc_lengths
        self.avgdl = float(doc_lengths.mean()) if self.N > 0 else 1.0
        print(f"  [BM25] Tokenized in {time.time()-t0:.1f}s | Vocab: {len(raw_index):,}")

        # Filter by df
        filtered = [(t, pl) for t, pl in raw_index.items() if len(pl) >= MIN_DF]
        filtered.sort(key=lambda x: -len(x[1]))
        filtered = filtered[:MAX_VOCAB]
        print(f"  [BM25] Vocab after filter: {len(filtered):,}")

        t2 = time.time()
        idf_values = []
        for tid, (token, pl) in enumerate(filtered):
            self.token2id[token] = tid
            df = len(pl)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            idf_values.append(idf)
            pos_arr = np.array([p for p, _ in pl], dtype=np.int32)
            tf_arr  = np.array([tf for _, tf in pl], dtype=np.float32)
            self.postings.append((pos_arr, tf_arr))

        self.idf = np.array(idf_values, dtype=np.float32)
        del raw_index, filtered; gc.collect()
        print(f"  [BM25] Index ready in {time.time()-t0:.1f}s total")

    def query_one(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        tokens = tokenize(text)
        if not tokens:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        scores: Dict[int, float] = {}
        for token in set(tokens):
            tid = self.token2id.get(token)
            if tid is None: continue
            idf = float(self.idf[tid])
            pos_arr, tf_arr = self.postings[tid]
            for i in range(len(pos_arr)):
                pos = int(pos_arr[i])
                tf  = float(tf_arr[i])
                dl  = float(self.doc_len[pos])
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[pos] = scores.get(pos, 0.0) + idf * num / den

        if not scores:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        positions  = np.fromiter(scores.keys(),   dtype=np.int32,   count=len(scores))
        score_arr  = np.fromiter(scores.values(), dtype=np.float32, count=len(scores))

        if len(score_arr) > TOP_K:
            idx = np.argpartition(score_arr, -TOP_K)[-TOP_K:]
            positions, score_arr = positions[idx], score_arr[idx]

        order = np.argsort(-score_arr)
        return positions[order], score_arr[order]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [BM25] Index cached → {path}")

    @staticmethod
    def load(path: str) -> 'BM25Index':
        with open(path, 'rb') as f:
            idx = pickle.load(f)
        return idx

# ============================================================
# INCREMENTAL CHUNK-WRITING QUERY LOOP
# ============================================================
def query_and_write(index: BM25Index,
                    s1_ids: np.ndarray,
                    s1_texts: List[str],
                    cand_ids: np.ndarray,
                    view_name: str,
                    out_path: str):
    """
    Query every S1 record, write results to parquet in CHUNK_WRITE-sized chunks.
    No large arrays ever sit in memory. Never returns big data structures.
    """
    n = len(s1_ids)
    writer = None
    schema = pa.schema([
        pa.field("s1_id",        pa.string()),
        pa.field("candidate_id", pa.string()),
        pa.field("view",         pa.string()),
        pa.field("rank",         pa.int16()),
        pa.field("score",        pa.float32()),
    ])

    buf_s1    = []
    buf_cand  = []
    buf_rank  = []
    buf_score = []

    t0 = time.time()

    def flush():
        nonlocal writer, buf_s1, buf_cand, buf_rank, buf_score
        if not buf_s1: return
        batch = pa.record_batch({
            "s1_id":        pa.array(buf_s1,    type=pa.string()),
            "candidate_id": pa.array(buf_cand,  type=pa.string()),
            "view":         pa.array([view_name] * len(buf_s1), type=pa.string()),
            "rank":         pa.array(buf_rank,  type=pa.int16()),
            "score":        pa.array(buf_score, type=pa.float32()),
        })
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema=batch.schema)
        writer.write_batch(batch)
        buf_s1, buf_cand, buf_rank, buf_score = [], [], [], []

    for i, (s1_id, text) in enumerate(zip(s1_ids, s1_texts)):
        positions, scores = index.query_one(text)

        for rank_idx, (pos, sc) in enumerate(zip(positions, scores), start=1):
            buf_s1.append(str(s1_id))
            buf_cand.append(str(cand_ids[pos]))
            buf_rank.append(rank_idx)
            buf_score.append(float(sc))

        if (i + 1) % CHUNK_WRITE == 0:
            flush()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (n - i - 1) / rate
            print(f"    [{view_name}] {i+1:,}/{n:,} | "
                  f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    flush()
    if writer: writer.close()

# ============================================================
# MAIN
# ============================================================
def load_view_texts(view_cols: List[str]):
    needed = ["entity_id"] + view_cols

    def safe_read(path):
        all_cols = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
        use = [c for c in needed if c in all_cols]
        return pd.read_csv(path, sep='\t', usecols=use, dtype=str).fillna("")

    s1 = safe_read(os.path.join(PROCESSED_DIR, "train_source1_norm.tsv"))
    s2 = safe_read(os.path.join(PROCESSED_DIR, "train_source2_norm.tsv"))
    s3 = safe_read(os.path.join(PROCESSED_DIR, "train_source3_norm.tsv"))

    s1_ids   = s1["entity_id"].values
    cand_ids = np.concatenate([s2["entity_id"].values, s3["entity_id"].values])

    s1_texts   = s1.apply(lambda r: build_text(r, view_cols), axis=1).tolist()
    cand_texts = (pd.concat([s2, s3], ignore_index=True)
                    .apply(lambda r: build_text(r, view_cols), axis=1)
                    .tolist())

    del s1, s2, s3; gc.collect()
    return s1_ids, cand_ids, s1_texts, cand_texts


def run_phase3_4():
    print(f"🚀 Starting Phase 3 & 4 | No multiprocessing IPC deadlocks!\n")

    for view_name, view_cols in VIEWS.items():
        out_path = os.path.join(OUT_DIR, f"bm25_{view_name}_top{TOP_K}.parquet")
        idx_path = os.path.join(CACHE_DIR, f"bm25_{view_name}_index.pkl")

        if os.path.exists(out_path):
            print(f"⏭️  [{view_name.upper()}] Checkpoint found — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"  [{view_name.upper()} VIEW]  cols: {view_cols}")
        print(f"{'='*55}")
        t_view = time.time()

        print("  Loading texts...")
        s1_ids, cand_ids, s1_texts, cand_texts = load_view_texts(view_cols)
        print(f"  Queries: {len(s1_ids):,}  |  Candidates: {len(cand_ids):,}")

        if os.path.exists(idx_path):
            print("  💾 Loading cached BM25 index...")
            bm25 = BM25Index.load(idx_path)
            del cand_texts; gc.collect()
        else:
            print("  🔨 Building BM25 index...")
            bm25 = BM25Index()
            bm25.build(cand_texts, cand_ids)
            bm25.save(idx_path)
            del cand_texts; gc.collect()

        print(f"\n  🔍 Querying {len(s1_ids):,} records → writing every {CHUNK_WRITE:,} rows...")
        t_q = time.time()
        query_and_write(bm25, s1_ids, s1_texts, cand_ids, view_name, out_path)
        print(f"  ✅ [{view_name.upper()}] Querying done in {time.time()-t_q:.1f}s")

        del bm25, s1_ids, s1_texts, cand_ids; gc.collect()
        print(f"  🎉 [{view_name.upper()}] Total: {time.time()-t_view:.1f}s")

    print("\n✅ All views complete. Phase 3 & 4 done!")


if __name__ == "__main__":
    run_phase3_4()
