"""
ECHO-ER Phase 3 & 4: OPTIMIZED Multi-View Sparse Retrieval
============================================================
Strategy: Inverted Index + BM25 scoring
- Never computes a full N x M matrix
- Only scores candidates that share at least 1 token with query
- Queries processed in parallel across all 4 CPU cores
- Checkpointed per view (resume safely if Kaggle times out)
- BM25 parameters tunable via config
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import List, Dict, Tuple

# ============================================================
# CONFIG
# ============================================================
TOP_K      = 30          # Candidates to retrieve per query
BM25_K1    = 1.5         # BM25 term saturation
BM25_B     = 0.75        # BM25 length normalization
MAX_VOCAB  = 500_000     # Cap vocabulary size
MIN_DF     = 2           # Token must appear in >= 2 docs

CACHE_DIR      = "/kaggle/working/cache"
OUT_DIR        = "/kaggle/working/retrieval"
PROCESSED_DIR  = "/kaggle/working/data/processed/train"

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
    if not text or pd.isna(text): return []
    return _RE_TOKEN.findall(text.lower())

def build_text(row: pd.Series, cols: List[str]) -> str:
    return " ".join(str(row.get(c, "")) for c in cols if not pd.isna(row.get(c, "")))

# ============================================================
# BM25 INVERTED INDEX
# ============================================================
class BM25Index:
    """
    Fast BM25 index backed by Python arrays.
    - Build: O(sum of document lengths)
    - Query: O(query_tokens x mean_posting_list_length)
    """
    def __init__(self, k1=BM25_K1, b=BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_ids = None          # np.array: position → entity_id
        self.doc_len = None          # np.array: position → token_count
        self.avgdl = 0.0
        self.N = 0
        # inverted index: token_id → np.array([pos, tf, ...])
        self.token2id: Dict[str, int] = {}
        self.postings: List[Tuple[np.ndarray, np.ndarray]] = []  # (positions, tfs)
        self.idf: np.ndarray = None  # per token

    def build(self, texts: List[str], doc_ids: np.ndarray):
        print("  [BM25] Tokenizing and counting term frequencies...")
        t0 = time.time()
        self.N = len(texts)
        self.doc_ids = doc_ids

        # First pass: compute doc lengths and raw term freqs per doc
        raw_index: Dict[str, List] = defaultdict(list)  # token → [(pos, tf)]
        doc_lengths = np.zeros(self.N, dtype=np.int32)

        for pos, text in enumerate(texts):
            tokens = tokenize(text)
            doc_lengths[pos] = len(tokens)
            if not tokens: continue
            # Count TF in this doc
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                raw_index[t].append((pos, tf))

        self.doc_len = doc_lengths
        self.avgdl = float(doc_lengths.mean()) if self.N > 0 else 1.0
        print(f"  [BM25] Tokenization done in {time.time()-t0:.1f}s. Vocab before filtering: {len(raw_index):,}")

        # Filter by MIN_DF and cap vocab
        t1 = time.time()
        filtered = [(t, postings) for t, postings in raw_index.items() if len(postings) >= MIN_DF]
        # Sort by df descending to drop least frequent if over MAX_VOCAB
        filtered.sort(key=lambda x: -len(x[1]))
        filtered = filtered[:MAX_VOCAB]
        print(f"  [BM25] Vocab after filtering: {len(filtered):,}  ({time.time()-t1:.1f}s)")

        # Build compact arrays
        t2 = time.time()
        idf_values = []
        for tid, (token, postings_list) in enumerate(filtered):
            self.token2id[token] = tid
            df = len(postings_list)
            # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            idf_values.append(idf)
            pos_arr = np.array([p for p, _ in postings_list], dtype=np.int32)
            tf_arr  = np.array([tf for _, tf in postings_list], dtype=np.float32)
            self.postings.append((pos_arr, tf_arr))

        self.idf = np.array(idf_values, dtype=np.float32)
        del raw_index, filtered; gc.collect()
        print(f"  [BM25] Index built in {time.time()-t2:.1f}s. Total build time: {time.time()-t0:.1f}s")

    def query_one(self, text: str, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        """Score all candidates matching query tokens. Return top_k (positions, scores)."""
        tokens = tokenize(text)
        if not tokens:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # Accumulate scores for matching candidates
        scores: Dict[int, float] = {}
        seen_tids = set()

        for token in set(tokens):
            tid = self.token2id.get(token)
            if tid is None or tid in seen_tids: continue
            seen_tids.add(tid)

            idf = float(self.idf[tid])
            pos_arr, tf_arr = self.postings[tid]

            for i in range(len(pos_arr)):
                pos = int(pos_arr[i])
                tf  = float(tf_arr[i])
                dl  = float(self.doc_len[pos])
                # BM25 score
                numerator   = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                delta = idf * numerator / denominator
                scores[pos] = scores.get(pos, 0.0) + delta

        if not scores:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # Fast top-K via argpartition
        positions = np.array(list(scores.keys()), dtype=np.int32)
        score_arr = np.array(list(scores.values()), dtype=np.float32)

        if len(score_arr) > top_k:
            idx = np.argpartition(score_arr, -top_k)[-top_k:]
            positions, score_arr = positions[idx], score_arr[idx]

        order = np.argsort(-score_arr)
        return positions[order], score_arr[order]

    def query_batch(self, texts: List[str], top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        """Query multiple texts, return (n_queries x top_k) arrays."""
        n = len(texts)
        all_pos    = np.full((n, top_k), -1, dtype=np.int32)
        all_scores = np.zeros((n, top_k), dtype=np.float32)
        for i, text in enumerate(texts):
            pos, scores = self.query_one(text, top_k)
            k = len(pos)
            if k > 0:
                all_pos[i, :k]    = pos[:k]
                all_scores[i, :k] = scores[:k]
        return all_pos, all_scores

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [BM25] Index saved to {path}")

    @staticmethod
    def load(path: str) -> 'BM25Index':
        with open(path, 'rb') as f:
            idx = pickle.load(f)
        print(f"  [BM25] Index loaded from {path}")
        return idx

# ============================================================
# PARALLEL QUERY WORKER
# ============================================================
def _query_worker(args):
    """Runs query_batch on a slice of queries. Used by ProcessPoolExecutor."""
    index_path, texts_slice, top_k = args
    idx = BM25Index.load(index_path)
    return idx.query_batch(texts_slice, top_k)

def parallel_query(index_path: str, query_texts: List[str],
                   top_k: int = TOP_K, num_workers: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distributes query_texts evenly across CPU cores.
    Each worker loads the cached index independently (avoids pickling large objects).
    """
    n = len(query_texts)
    chunk_size = math.ceil(n / num_workers)
    chunks = [query_texts[i:i+chunk_size] for i in range(0, n, chunk_size)]

    all_pos    = np.full((n, top_k), -1, dtype=np.int32)
    all_scores = np.zeros((n, top_k), dtype=np.float32)

    args = [(index_path, chunk, top_k) for chunk in chunks]
    start = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for (pos_chunk, score_chunk) in pool.map(_query_worker, args):
            end = start + pos_chunk.shape[0]
            all_pos[start:end]    = pos_chunk
            all_scores[start:end] = score_chunk
            start = end
            print(f"    ✅ Worker done  [{end:,}/{n:,}]")

    return all_pos, all_scores

# ============================================================
# MAIN PIPELINE
# ============================================================
def load_view_texts(view_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load S1 query texts and candidate texts for the given view columns."""
    needed = ["entity_id"] + [c for c in view_cols if c != "entity_id"]

    def safe_read(path):
        all_cols = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
        usecols = [c for c in needed if c in all_cols]
        return pd.read_csv(path, sep='\t', usecols=usecols, dtype=str).fillna("")

    s1 = safe_read(os.path.join(PROCESSED_DIR, "train_source1_norm.tsv"))
    s2 = safe_read(os.path.join(PROCESSED_DIR, "train_source2_norm.tsv"))
    s3 = safe_read(os.path.join(PROCESSED_DIR, "train_source3_norm.tsv"))

    s1_ids = s1["entity_id"].values
    cand_ids = np.concatenate([s2["entity_id"].values, s3["entity_id"].values])

    s1_texts   = s1.apply(lambda r: build_text(r, view_cols), axis=1).tolist()
    cand_texts = (pd.concat([s2, s3], ignore_index=True)
                    .apply(lambda r: build_text(r, view_cols), axis=1)
                    .tolist())

    del s1, s2, s3; gc.collect()
    return s1_ids, cand_ids, s1_texts, cand_texts

def run_phase3_4():
    num_cores = multiprocessing.cpu_count()
    print(f"🚀 Starting Phase 3 & 4 | {num_cores} CPU cores\n")

    for view_name, view_cols in VIEWS.items():
        out_path   = os.path.join(OUT_DIR, f"bm25_{view_name}_top{TOP_K}.parquet")
        idx_path   = os.path.join(CACHE_DIR, f"bm25_{view_name}_index.pkl")

        if os.path.exists(out_path):
            print(f"⏭️  [{view_name.upper()} VIEW] Checkpoint found — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"  [{view_name.upper()} VIEW]  cols: {view_cols}")
        print(f"{'='*55}")
        t_view = time.time()

        # --- Load texts ---
        print("  Loading texts...")
        s1_ids, cand_ids, s1_texts, cand_texts = load_view_texts(view_cols)
        print(f"  Queries: {len(s1_ids):,}  |  Candidates: {len(cand_ids):,}")

        # --- Build / Load Index ---
        if os.path.exists(idx_path):
            print("  💾 Loading cached BM25 index...")
            bm25 = BM25Index.load(idx_path)
        else:
            print("  🔨 Building BM25 index on candidate corpus...")
            bm25 = BM25Index(k1=BM25_K1, b=BM25_B)
            bm25.build(cand_texts, cand_ids)
            bm25.save(idx_path)

        del cand_texts; gc.collect()

        # --- Query in Parallel ---
        print(f"  🔍 Querying Top-{TOP_K} across {num_cores} cores...")
        t_q = time.time()
        pos_matrix, score_matrix = parallel_query(idx_path, s1_texts, TOP_K, num_workers=num_cores)
        print(f"  Querying done in {time.time()-t_q:.1f}s")

        del s1_texts, bm25; gc.collect()

        # --- Flatten & Save ---
        print("  💾 Saving results as parquet...")
        n = len(s1_ids)
        q_ids_rep  = np.repeat(s1_ids, TOP_K)
        ranks_flat = np.tile(np.arange(1, TOP_K + 1), n)
        scores_flat = score_matrix.flatten()

        # Map position indices → actual candidate entity_ids
        flat_pos    = pos_matrix.flatten()
        valid_mask  = flat_pos >= 0
        cand_mapped = np.full(len(flat_pos), "", dtype=object)
        cand_mapped[valid_mask] = cand_ids[flat_pos[valid_mask]]

        df_out = pd.DataFrame({
            "s1_id":       q_ids_rep,
            "candidate_id": cand_mapped,
            "view":        view_name,
            "rank":        ranks_flat,
            "score":       scores_flat,
        })
        # Drop empty (no-match) rows
        df_out = df_out[df_out["candidate_id"] != ""].reset_index(drop=True)
        df_out.to_parquet(out_path, index=False)

        print(f"  ✅ [{view_name.upper()} VIEW] Done | {len(df_out):,} pairs | {time.time()-t_view:.1f}s total")
        del df_out, q_ids_rep, cand_mapped, pos_matrix, score_matrix; gc.collect()

    print("\n🎉 All views checkpointed. Phase 3 & 4 complete!")

if __name__ == "__main__":
    run_phase3_4()
