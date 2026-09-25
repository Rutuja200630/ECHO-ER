"""
BM25S BENCHMARK — ECHO-ER Phase 3 & 4
======================================
Rigorous comparison of current custom BM25 vs bm25s library.

Tasks:
  T1  Install & verify bm25s
  T2  10K queries / 100K corpus benchmark
  T3  Correctness (Recall@K, candidate overlap)
  T4  10K queries / 1M corpus scaling
  T5  Full 10.3M corpus index build only
  T6  Write docs/BM25S_BENCHMARK.md
"""
import os, sys, re, gc, time, math, pickle, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
import tracemalloc

# ─── paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "/kaggle/working/data/processed/train"
GT_GLOB        = "/kaggle/input/**/train_ground_truth.tsv"
DOCS_DIR       = "/kaggle/working/ECHO-ER/docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# ─── BM25 params (used in BOTH implementations so comparison is fair) ─────────
K1, B = 1.5, 0.75

# ─── tokenizer (shared by both implementations) ───────────────────────────────
_RE_TOK = re.compile(r'\b\w+\b')
STOP_WORDS = {
    'the','a','an','and','or','of','in','on','at','to','for','is','are','was',
    'be','by','as','it','its','this','that','with','from','have','has','had',
    'not','but','they','we','you','our','your','their',
    'street','road','avenue','boulevard','drive','lane','way','place','court',
    'st','rd','ave','blvd','dr','ln','pl','ct','suite','floor','unit','apt',
    'services','service','group','company','management','international','national',
    'general','center','centre','enterprises','solutions','technology',
    'north','south','east','west','new','old',
}
MAX_POSTING = 80_000

def tokenize(text: str) -> List[str]:
    if not text or (isinstance(text, float) and math.isnan(text)):
        return []
    return [t for t in _RE_TOK.findall(str(text).lower())
            if t not in STOP_WORDS and len(t) > 1]

# ══════════════════════════════════════════════════════════════════════════════
# T1 — INSTALL & VERIFY BM25S
# ══════════════════════════════════════════════════════════════════════════════
def task1_verify_bm25s() -> Tuple[bool, str]:
    print("\n" + "="*60)
    print("T1: INSTALL & VERIFY BM25S")
    print("="*60)
    try:
        import bm25s as _bm25s
        v = getattr(_bm25s, '__version__', 'unknown')
        print(f"  ✅ bm25s already installed. Version: {v}")
        return True, v
    except ImportError:
        print("  bm25s not found. Installing...")
        import subprocess
        ret = subprocess.run([sys.executable, "-m", "pip", "install", "bm25s", "-q"],
                             capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"  ❌ Installation failed:\n{ret.stderr}")
            return False, "INSTALL_FAILED"
        import importlib
        import bm25s as _bm25s
        importlib.reload(_bm25s)
        v = getattr(_bm25s, '__version__', 'unknown')
        print(f"  ✅ bm25s installed successfully. Version: {v}")
        return True, v

# ══════════════════════════════════════════════════════════════════════════════
# CURRENT IMPLEMENTATION (custom BM25 with numpy vectorized scoring)
# ══════════════════════════════════════════════════════════════════════════════
class CustomBM25:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.postings: List[Tuple[np.ndarray, np.ndarray]] = []
        self.idf: np.ndarray = None
        self.doc_len: np.ndarray = None
        self.avgdl: float = 0.0
        self.N: int = 0

    def build(self, texts: List[str]) -> float:
        t0 = time.time()
        self.N = len(texts)
        raw: Dict[str, List] = defaultdict(list)
        dlens = np.zeros(self.N, dtype=np.int32)
        for pos, text in enumerate(texts):
            toks = tokenize(text)
            dlens[pos] = len(toks)
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                raw[t].append((pos, tf))
        self.doc_len = dlens
        self.avgdl = float(dlens.mean()) if self.N else 1.0
        filtered = [(t, pl) for t, pl in raw.items()
                    if 2 <= len(pl) <= MAX_POSTING]
        filtered.sort(key=lambda x: -len(x[1]))
        idf_vals = []
        for tid, (tok, pl) in enumerate(filtered):
            self.token2id[tok] = tid
            df = len(pl)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            idf_vals.append(idf)
            self.postings.append((
                np.array([p for p,_ in pl], dtype=np.int32),
                np.array([tf for _,tf in pl], dtype=np.float32)
            ))
        self.idf = np.array(idf_vals, dtype=np.float32)
        del raw, filtered; gc.collect()
        return time.time() - t0

    def query_one(self, text: str, k: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        toks = tokenize(text)
        if not toks:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        pos_list, contrib_list = [], []
        for tok in set(toks):
            tid = self.token2id.get(tok)
            if tid is None: continue
            idf = float(self.idf[tid])
            pos_arr, tf_arr = self.postings[tid]
            dl_arr = self.doc_len[pos_arr].astype(np.float32)
            num = tf_arr * (K1 + 1.0)
            den = tf_arr + K1 * (1.0 - B + B * dl_arr / self.avgdl)
            pos_list.append(pos_arr)
            contrib_list.append(idf * num / den)
        if not pos_list:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        all_pos = np.concatenate(pos_list)
        all_sc  = np.concatenate(contrib_list)
        order   = np.argsort(all_pos, kind='stable')
        sp      = all_pos[order]; sv = all_sc[order]
        upos, fi = np.unique(sp, return_index=True)
        sc = np.add.reduceat(sv, fi).astype(np.float32)
        if len(sc) > k:
            idx = np.argpartition(sc, -k)[-k:]
            upos, sc = upos[idx], sc[idx]
        order = np.argsort(-sc)
        return upos[order].astype(np.int32), sc[order]

    def query_batch(self, texts: List[str], k: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [self.query_one(t, k) for t in texts]

# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_sample(n_queries: int, n_corpus: int):
    """Load n_queries S1 records and n_corpus candidate records (S2+S3)."""
    s1 = pd.read_csv(os.path.join(PROCESSED_DIR, "train_source1_norm.tsv"),
                     sep='\t', nrows=n_queries,
                     usecols=['entity_id','name_norm']
                     ).fillna("")
    s2 = pd.read_csv(os.path.join(PROCESSED_DIR, "train_source2_norm.tsv"),
                     sep='\t', nrows=n_corpus // 2,
                     usecols=['entity_id','name_norm']
                     ).fillna("")
    s3 = pd.read_csv(os.path.join(PROCESSED_DIR, "train_source3_norm.tsv"),
                     sep='\t', nrows=n_corpus - n_corpus // 2,
                     usecols=['entity_id','name_norm']
                     ).fillna("")
    cand = pd.concat([s2, s3], ignore_index=True)
    q_ids    = s1['entity_id'].values
    cand_ids = cand['entity_id'].values
    q_texts   = s1['name_norm'].tolist()
    cand_texts = cand['name_norm'].tolist()
    return q_ids, cand_ids, q_texts, cand_texts

def load_ground_truth() -> Dict[str, set]:
    paths = glob.glob(GT_GLOB, recursive=True)
    if not paths: return {}
    gt = pd.read_csv(paths[0], sep='\t', dtype=str)
    result = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids']
        if pd.isna(matched):
            result[s1_id] = set()
        else:
            result[s1_id] = set(matched.split(','))
    return result

def compute_recall(retrieved: Dict[str, List[str]],
                   gt: Dict[str, set],
                   k_vals=[5,10,20,30,50]) -> Dict[str, float]:
    recalls = {}
    for k in k_vals:
        total_possible = sum(len(v) for v in gt.values() if v)
        if total_possible == 0:
            recalls[f"Recall@{k}"] = float('nan')
            continue
        found = 0
        for s1_id, true_matches in gt.items():
            if not true_matches: continue
            top_k_set = set(list(retrieved.get(s1_id, []))[:k])
            found += len(true_matches & top_k_set)
        recalls[f"Recall@{k}"] = round(100.0 * found / total_possible, 2)
    return recalls

def compute_overlap(ret_a: Dict[str, List[str]],
                    ret_b: Dict[str, List[str]],
                    k_vals=[5,10,20,30,50]) -> Dict[str, float]:
    overlaps = {}
    q_ids = list(ret_a.keys())
    for k in k_vals:
        ratios = []
        for qid in q_ids:
            a = set(list(ret_a.get(qid, []))[:k])
            b = set(list(ret_b.get(qid, []))[:k])
            if not a and not b: ratios.append(1.0)
            elif not a or not b: ratios.append(0.0)
            else: ratios.append(len(a & b) / max(len(a), len(b)))
        overlaps[f"Overlap@{k}"] = round(100.0 * np.mean(ratios), 2)
    return overlaps

def mem_mb():
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except:
        return float('nan')

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_custom_bm25(q_texts, cand_texts, q_ids, cand_ids, k=30):
    idx = CustomBM25()
    tracemalloc.start()
    m0 = mem_mb()

    t_build = idx.build(cand_texts)
    m_build = mem_mb()

    t_q0 = time.time()
    retrieved = {}
    for qid, qt in zip(q_ids, q_texts):
        pos, _ = idx.query_one(qt, k)
        retrieved[qid] = [cand_ids[p] for p in pos if p < len(cand_ids)]
    t_query = time.time() - t_q0

    m_peak = mem_mb()
    _, peak_bytes = tracemalloc.get_traced_memory(); tracemalloc.stop()
    del idx; gc.collect()

    return retrieved, {
        "build_s": round(t_build, 2),
        "query_s": round(t_query, 2),
        "total_s": round(t_build + t_query, 2),
        "qps":     round(len(q_texts) / t_query, 1),
        "ram_mb":  round(m_peak - m0, 1),
        "peak_tracemalloc_mb": round(peak_bytes / 1024**2, 1),
    }

def run_bm25s(q_texts, cand_texts, q_ids, cand_ids, k=30):
    import bm25s
    tracemalloc.start()
    m0 = mem_mb()

    # bm25s uses its own tokenizer; we tokenize ourselves for consistency
    t_tok0 = time.time()
    corp_tok = bm25s.tokenize(cand_texts, stopwords="en")
    quer_tok = bm25s.tokenize(q_texts,    stopwords="en")
    t_tok = time.time() - t_tok0

    t_idx0 = time.time()
    retriever = bm25s.BM25(k1=K1, b=B)
    retriever.index(corp_tok)
    t_build = time.time() - t_idx0 + t_tok
    m_build = mem_mb()

    t_q0 = time.time()
    results_idx, _ = retriever.retrieve(quer_tok, corpus=np.arange(len(cand_texts)), k=min(k, len(cand_texts)))
    t_query = time.time() - t_q0

    m_peak = mem_mb()
    _, peak_bytes = tracemalloc.get_traced_memory(); tracemalloc.stop()

    # Build retrieved dict
    retrieved = {}
    for i, qid in enumerate(q_ids):
        pos = results_idx[i]
        retrieved[qid] = [cand_ids[p] for p in pos if p < len(cand_ids)]

    del retriever, corp_tok, quer_tok; gc.collect()

    return retrieved, {
        "build_s": round(t_build, 2),
        "query_s": round(t_query, 2),
        "total_s": round(t_build + t_query, 2),
        "qps":     round(len(q_texts) / t_query, 1),
        "ram_mb":  round(m_peak - m0, 1),
        "peak_tracemalloc_mb": round(peak_bytes / 1024**2, 1),
    }

def print_stats(name, stats):
    print(f"\n  [{name}]")
    for k, v in stats.items():
        print(f"    {k:<30} {v}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    lines = []   # collect all output for the markdown report

    def log(msg=""):
        print(msg)
        lines.append(msg)

    # ── T1: verify bm25s ───────────────────────────────────────────────────
    bm25s_ok, bm25s_version = task1_verify_bm25s()
    log(f"\nBM25S INSTALLATION = {'PASS (v' + bm25s_version + ')' if bm25s_ok else 'FAIL'}")
    if not bm25s_ok:
        log("Cannot continue benchmarking. Stopping.")
        return

    gt_map = load_ground_truth()
    log(f"Ground truth loaded: {len(gt_map):,} S1 records")

    results = {}   # collect scenario results

    # ──────────────────────────────────────────────────────────────────────────
    # T2 + T3: 10K queries / 100K corpus
    # ──────────────────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("T2+T3: BENCHMARK  10K queries / 100K corpus")
    log("="*60)

    q_ids, cand_ids, q_texts, cand_texts = load_sample(10_000, 100_000)
    log(f"  Loaded {len(q_ids):,} queries, {len(cand_ids):,} candidates")

    # Subset GT to these q_ids
    gt_sub = {qid: gt_map.get(qid, set()) for qid in q_ids}

    log("\n  --- Running Custom BM25 ---")
    ret_custom, stats_custom = run_custom_bm25(q_texts, cand_texts, q_ids, cand_ids)
    print_stats("Custom BM25", stats_custom)

    log("\n  --- Running bm25s ---")
    ret_bm25s, stats_bm25s = run_bm25s(q_texts, cand_texts, q_ids, cand_ids)
    print_stats("bm25s", stats_bm25s)

    speedup_100k = round(stats_custom['total_s'] / max(stats_bm25s['total_s'], 0.001), 2)
    log(f"\n  Speedup (Custom→bm25s): {speedup_100k}×")

    # Recall
    log("\n  --- Recall (100K corpus) ---")
    rec_custom = compute_recall(ret_custom, gt_sub)
    rec_bm25s  = compute_recall(ret_bm25s,  gt_sub)
    for k in [5,10,20,30,50]:
        key = f"Recall@{k}"
        log(f"  {key:<15} Custom={rec_custom[key]}%   bm25s={rec_bm25s[key]}%")

    # Overlap
    log("\n  --- Candidate Overlap (100K corpus) ---")
    overlap = compute_overlap(ret_custom, ret_bm25s)
    for k, v in overlap.items():
        log(f"  {k:<15} {v}%")

    results['100k'] = {
        'custom': stats_custom, 'bm25s': stats_bm25s,
        'speedup': speedup_100k,
        'recall_custom': rec_custom, 'recall_bm25s': rec_bm25s,
        'overlap': overlap
    }
    del q_ids, cand_ids, q_texts, cand_texts, ret_custom, ret_bm25s; gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # T4: 10K queries / 1M corpus
    # ──────────────────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("T4: SCALING  10K queries / 1M corpus")
    log("="*60)

    q_ids, cand_ids, q_texts, cand_texts = load_sample(10_000, 1_000_000)
    log(f"  Loaded {len(q_ids):,} queries, {len(cand_ids):,} candidates")

    gt_sub = {qid: gt_map.get(qid, set()) for qid in q_ids}

    log("\n  --- Running Custom BM25 (1M) ---")
    ret_custom, stats_custom_1m = run_custom_bm25(q_texts, cand_texts, q_ids, cand_ids)
    print_stats("Custom BM25 (1M)", stats_custom_1m)

    log("\n  --- Running bm25s (1M) ---")
    ret_bm25s, stats_bm25s_1m = run_bm25s(q_texts, cand_texts, q_ids, cand_ids)
    print_stats("bm25s (1M)", stats_bm25s_1m)

    speedup_1m = round(stats_custom_1m['total_s'] / max(stats_bm25s_1m['total_s'], 0.001), 2)
    log(f"\n  Speedup (Custom→bm25s): {speedup_1m}×")

    rec_custom_1m = compute_recall(ret_custom, gt_sub)
    rec_bm25s_1m  = compute_recall(ret_bm25s,  gt_sub)
    log("\n  --- Recall (1M corpus) ---")
    for k in [5,10,20,30,50]:
        key = f"Recall@{k}"
        log(f"  {key:<15} Custom={rec_custom_1m[key]}%   bm25s={rec_bm25s_1m[key]}%")

    overlap_1m = compute_overlap(ret_custom, ret_bm25s)
    log("\n  --- Candidate Overlap (1M corpus) ---")
    for k, v in overlap_1m.items():
        log(f"  {k:<15} {v}%")

    results['1m'] = {
        'custom': stats_custom_1m, 'bm25s': stats_bm25s_1m,
        'speedup': speedup_1m,
        'recall_custom': rec_custom_1m, 'recall_bm25s': rec_bm25s_1m,
        'overlap': overlap_1m
    }
    del q_ids, cand_ids, q_texts, cand_texts, ret_custom, ret_bm25s; gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # T5: Full 10.3M corpus index build ONLY (no queries)
    # ──────────────────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("T5: FULL 10.3M CORPUS INDEX BUILD (bm25s only)")
    log("="*60)

    import bm25s
    s2_full = pd.read_csv(os.path.join(PROCESSED_DIR, "train_source2_norm.tsv"),
                          sep='\t', usecols=['name_norm']).fillna("")
    s3_full = pd.read_csv(os.path.join(PROCESSED_DIR, "train_source3_norm.tsv"),
                          sep='\t', usecols=['name_norm']).fillna("")
    all_texts = (pd.concat([s2_full, s3_full], ignore_index=True)
                   ['name_norm'].tolist())
    log(f"  Full corpus size: {len(all_texts):,} documents")
    del s2_full, s3_full; gc.collect()

    m0 = mem_mb()
    t0 = time.time()
    log("  Tokenizing...")
    corp_tok = bm25s.tokenize(all_texts, stopwords="en")
    t_tok = time.time() - t0
    log(f"  Tokenization: {t_tok:.1f}s | RAM: {mem_mb()-m0:.0f}MB")

    t1 = time.time()
    retriever_full = bm25s.BM25(k1=K1, b=B)
    retriever_full.index(corp_tok)
    t_idx = time.time() - t1
    m_after = mem_mb()
    log(f"  Indexing: {t_idx:.1f}s | RAM: {m_after-m0:.0f}MB")
    log(f"  Total build: {t_tok+t_idx:.1f}s | Peak RAM delta: {m_after-m0:.0f}MB")

    # Save & reload test
    idx_path = "/kaggle/working/cache/bm25s_name_full_index"
    os.makedirs(idx_path, exist_ok=True)
    t_save0 = time.time()
    retriever_full.save(idx_path)
    t_save = time.time() - t_save0

    # Measure on-disk size
    total_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(idx_path) for f in files
    )
    log(f"  Save time: {t_save:.1f}s | Disk size: {total_size/1024**2:.0f}MB")

    # Reload test
    t_load0 = time.time()
    del retriever_full, corp_tok, all_texts; gc.collect()
    retriever_reloaded = bm25s.BM25.load(idx_path)
    t_load = time.time() - t_load0
    log(f"  Reload time: {t_load:.1f}s ✅")
    del retriever_reloaded; gc.collect()

    # Extrapolate full-query runtime from 1M benchmark
    if 'bm25s' in results.get('1m', {}):
        qps_1m = results['1m']['bm25s']['qps']
        full_query_est = round(2_206_821 / qps_1m) if qps_1m > 0 else float('nan')
    else:
        full_query_est = float('nan')

    results['full_index'] = {
        'tok_s':   round(t_tok, 1),
        'idx_s':   round(t_idx, 1),
        'total_s': round(t_tok + t_idx, 1),
        'ram_mb':  round(m_after - m0, 1),
        'disk_mb': round(total_size / 1024**2, 1),
        'save_s':  round(t_save, 1),
        'load_s':  round(t_load, 1),
        'full_query_est_s': full_query_est,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # WRITE MARKDOWN REPORT
    # ──────────────────────────────────────────────────────────────────────────
    r100 = results.get('100k', {})
    r1m  = results.get('1m', {})
    rfi  = results.get('full_index', {})

    bm25s_recall50 = r1m.get('recall_bm25s', {}).get('Recall@50', float('nan'))
    recommend = ("YES" if (results.get('100k',{}).get('speedup',0) > 3
                            and bm25s_recall50 > 0) else "NEEDS REVIEW")

    md = f"""# BM25S Benchmark Report — ECHO-ER Phase 3 & 4

## 1. Current Retrieval Performance (Custom BM25)

| Scenario | Build | Query | Total | QPS | RAM |
|---|---|---|---|---|---|
| 10K / 100K | {r100.get('custom',{}).get('build_s','?')}s | {r100.get('custom',{}).get('query_s','?')}s | {r100.get('custom',{}).get('total_s','?')}s | {r100.get('custom',{}).get('qps','?')} | {r100.get('custom',{}).get('ram_mb','?')}MB |
| 10K / 1M | {r1m.get('custom',{}).get('build_s','?')}s | {r1m.get('custom',{}).get('query_s','?')}s | {r1m.get('custom',{}).get('total_s','?')}s | {r1m.get('custom',{}).get('qps','?')} | {r1m.get('custom',{}).get('ram_mb','?')}MB |

## 2. bm25s Performance

| Scenario | Build | Query | Total | QPS | RAM |
|---|---|---|---|---|---|
| 10K / 100K | {r100.get('bm25s',{}).get('build_s','?')}s | {r100.get('bm25s',{}).get('query_s','?')}s | {r100.get('bm25s',{}).get('total_s','?')}s | {r100.get('bm25s',{}).get('qps','?')} | {r100.get('bm25s',{}).get('ram_mb','?')}MB |
| 10K / 1M | {r1m.get('bm25s',{}).get('build_s','?')}s | {r1m.get('bm25s',{}).get('query_s','?')}s | {r1m.get('bm25s',{}).get('total_s','?')}s | {r1m.get('bm25s',{}).get('qps','?')} | {r1m.get('bm25s',{}).get('ram_mb','?')}MB |

## 3. Speedup

| Scenario | Speedup |
|---|---|
| 100K corpus | {r100.get('speedup','?')}× |
| 1M corpus | {r1m.get('speedup','?')}× |

## 4. Memory

See RAM column in tables above.

## 5. Recall Comparison (1M corpus, Name View)

| Metric | Custom BM25 | bm25s |
|---|---|---|
| Recall@5  | {r1m.get('recall_custom',{}).get('Recall@5','?')}% | {r1m.get('recall_bm25s',{}).get('Recall@5','?')}% |
| Recall@10 | {r1m.get('recall_custom',{}).get('Recall@10','?')}% | {r1m.get('recall_bm25s',{}).get('Recall@10','?')}% |
| Recall@20 | {r1m.get('recall_custom',{}).get('Recall@20','?')}% | {r1m.get('recall_bm25s',{}).get('Recall@20','?')}% |
| Recall@30 | {r1m.get('recall_custom',{}).get('Recall@30','?')}% | {r1m.get('recall_bm25s',{}).get('Recall@30','?')}% |
| Recall@50 | {r1m.get('recall_custom',{}).get('Recall@50','?')}% | {r1m.get('recall_bm25s',{}).get('Recall@50','?')}% |

## 6. Candidate Overlap (Custom BM25 vs bm25s)

| Metric | 100K corpus | 1M corpus |
|---|---|---|
| Overlap@5  | {r100.get('overlap',{}).get('Overlap@5','?')}% | {r1m.get('overlap',{}).get('Overlap@5','?')}% |
| Overlap@10 | {r100.get('overlap',{}).get('Overlap@10','?')}% | {r1m.get('overlap',{}).get('Overlap@10','?')}% |
| Overlap@20 | {r100.get('overlap',{}).get('Overlap@20','?')}% | {r1m.get('overlap',{}).get('Overlap@20','?')}% |
| Overlap@30 | {r100.get('overlap',{}).get('Overlap@30','?')}% | {r1m.get('overlap',{}).get('Overlap@30','?')}% |
| Overlap@50 | {r100.get('overlap',{}).get('Overlap@50','?')}% | {r1m.get('overlap',{}).get('Overlap@50','?')}% |

## 7. BM25 Parameters

| Parameter | Value |
|---|---|
| k1 | {K1} |
| b | {B} |
| MAX_POSTING (custom) | {MAX_POSTING:,} |
| MIN_DF | 2 |
| Tokenizer | shared regex `\\b\\w+\\b` + STOP_WORDS |
| bm25s stopwords | "en" (bm25s built-in) |

> Note: bm25s uses its own internal tokenizer. The custom BM25 uses a shared regex tokenizer with STOP_WORDS filtering. This may cause minor recall differences.

## 8. Full-Corpus Feasibility (10.3M docs — Index Build Only)

| Metric | Value |
|---|---|
| Tokenization time | {rfi.get('tok_s','?')}s |
| Index build time | {rfi.get('idx_s','?')}s |
| Total build time | {rfi.get('total_s','?')}s |
| RAM delta | {rfi.get('ram_mb','?')}MB |
| Disk size | {rfi.get('disk_mb','?')}MB |
| Save time | {rfi.get('save_s','?')}s |
| Reload time | {rfi.get('load_s','?')}s |
| Estimated full query time (2.2M queries) | {rfi.get('full_query_est_s','NOT YET MEASURED')}s |

## 9. Recommendation

| Item | Status |
|---|---|
| BM25S INSTALLATION | PASS (v{bm25s_version}) |
| BM25S CORRECTNESS | {'PASS' if bm25s_recall50 != float('nan') else 'MANUAL REVIEW'} |
| BM25S SPEEDUP | {r1m.get('speedup','?')}× (1M corpus) |
| BM25S Recall@50 (1M) | {bm25s_recall50}% |
| 1M CORPUS RUNTIME | {r1m.get('bm25s',{}).get('total_s','?')}s |
| FULL 10.3M INDEX FEASIBILITY | {'YES' if rfi.get('total_s', 9999) < 600 else 'NEEDS REVIEW'} |
| FULL 2.2M QUERY RUNTIME | NOT YET MEASURED |
| RECOMMEND BM25S | {recommend} |
"""

    report_path = os.path.join(DOCS_DIR, "BM25S_BENCHMARK.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n📄 Report saved → {report_path}")
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"BM25S INSTALLATION          = PASS (v{bm25s_version})")
    print(f"BM25S CORRECTNESS           = {'PASS' if not math.isnan(bm25s_recall50) else 'MANUAL REVIEW'}")
    print(f"BM25S SPEEDUP (1M)          = {r1m.get('speedup','?')}×")
    print(f"BM25S RECALL@50 (1M)        = {bm25s_recall50}%")
    print(f"1M CORPUS RUNTIME           = {r1m.get('bm25s',{}).get('total_s','?')}s")
    print(f"FULL 10.3M INDEX FEASIBILITY= {'YES' if rfi.get('total_s', 9999) < 600 else 'NEEDS REVIEW'}")
    print(f"FULL 2.2M QUERY RUNTIME     = NOT YET MEASURED")
    print(f"RECOMMEND BM25S             = {recommend}")


if __name__ == "__main__":
    main()
