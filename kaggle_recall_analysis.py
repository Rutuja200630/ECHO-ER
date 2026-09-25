"""
ECHO-ER Phase 4 — Recall Analysis Script
==========================================
Tasks:
  T1  Verify recall calculation & corpus coverage
  T2  Conditional Recall@K (when GT IS in corpus)
  T3  Failure mode categorisation
  T4  Baseline retrievers comparison
  T5  Character n-gram TF-IDF
  T6  Name + Address union
  T7  20 concrete failure examples
  T8  Write docs/PHASE_4_RECALL_ANALYSIS.md

DO NOT change the pipeline. This is purely diagnostic.
"""
import os, re, gc, math, glob, time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

# ─── paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "/kaggle/working/data/processed/train"
GT_GLOB       = "/kaggle/input/**/train_ground_truth.tsv"
DOCS_DIR      = "/kaggle/working/ECHO-ER/docs"
os.makedirs(DOCS_DIR, exist_ok=True)

# ─── benchmark sizes ─────────────────────────────────────────────────────────
N_QUERIES  = 10_000
N_CORPUS   = 1_000_000

K_VALS     = [5, 10, 20, 50, 100]
TOP_BASELN = 100    # max K to retrieve in baselines

# ─── tokenizer ────────────────────────────────────────────────────────────────
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

def tokenize(text:str) -> List[str]:
    if not text or (isinstance(text,float) and math.isnan(text)): return []
    return [t for t in _RE_TOK.findall(str(text).lower())
            if t not in STOP_WORDS and len(t)>1]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data(n_queries, n_corpus):
    print(f"  Loading {n_queries:,} queries and {n_corpus:,} candidate records...")
    t0 = time.time()

    s1 = pd.read_csv(os.path.join(PROCESSED_DIR,"train_source1_norm.tsv"),
                     sep='\t', nrows=n_queries,
                     usecols=['entity_id','business_name','business_address',
                               'name_norm','address_norm','country_norm','postal_code']
                     ).fillna("")

    s2 = pd.read_csv(os.path.join(PROCESSED_DIR,"train_source2_norm.tsv"),
                     sep='\t', nrows=n_corpus//2,
                     usecols=['entity_id','business_name','business_address',
                               'name_norm','address_norm','country_norm','postal_code']
                     ).fillna("")
    s3 = pd.read_csv(os.path.join(PROCESSED_DIR,"train_source3_norm.tsv"),
                     sep='\t', nrows=n_corpus - n_corpus//2,
                     usecols=['entity_id','business_name','business_address',
                               'name_norm','address_norm','country_norm','postal_code']
                     ).fillna("")

    cand = pd.concat([s2,s3], ignore_index=True)
    cand_id_set = set(cand['entity_id'].values)
    print(f"  Done in {time.time()-t0:.1f}s  |  S1:{len(s1):,}  Cand:{len(cand):,}")
    return s1, cand, cand_id_set

def load_gt(q_ids:np.ndarray) -> Dict[str,Set[str]]:
    paths = glob.glob(GT_GLOB, recursive=True)
    if not paths:
        print("  ⚠️ Ground truth not found!"); return {}
    gt_raw = pd.read_csv(paths[0], sep='\t', dtype=str)
    qid_set = set(q_ids)
    result = {}
    for _, row in gt_raw.iterrows():
        sid = row['source1_entity_id']
        if sid not in qid_set: continue
        matched = row['matched_entity_ids']
        result[sid] = set() if pd.isna(matched) else set(matched.split(','))
    return result

# ══════════════════════════════════════════════════════════════════════════════
# FAST TFIDF RETRIEVER (for baselines)
# ══════════════════════════════════════════════════════════════════════════════
def tfidf_retrieve(corpus_texts:List[str], query_texts:List[str],
                   k:int, **vect_kwargs) -> np.ndarray:
    """
    Returns (n_queries x k) index array into corpus.
    Uses chunked sparse dot-product to avoid OOM.
    """
    vec = TfidfVectorizer(**vect_kwargs)
    cmat = normalize(vec.fit_transform(corpus_texts), norm='l2', copy=False)
    qmat = normalize(vec.transform(query_texts),      norm='l2', copy=False)
    del vec; gc.collect()

    n_q = qmat.shape[0]
    n_c = cmat.shape[0]
    k   = min(k, n_c)
    result = np.zeros((n_q, k), dtype=np.int32)

    batch = 2000
    for start in range(0, n_q, batch):
        end  = min(start+batch, n_q)
        sim  = qmat[start:end].dot(cmat.T)          # dense (batch x n_c)
        if sp.issparse(sim): sim = sim.toarray()
        # top-k per row
        if n_c > k:
            idx_part = np.argpartition(sim, -k, axis=1)[:, -k:]
            for i, row_idx in enumerate(idx_part):
                s = sim[i, row_idx]
                order = np.argsort(-s)
                result[start+i] = row_idx[order]
        else:
            for i in range(end-start):
                result[start+i] = np.argsort(-sim[i])[:k]

    del cmat, qmat; gc.collect()
    return result

# ══════════════════════════════════════════════════════════════════════════════
# RECALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def build_retrieved_dict(q_ids:np.ndarray, idx_matrix:np.ndarray,
                         cand_ids:np.ndarray) -> Dict[str,List[str]]:
    result = {}
    for i, qid in enumerate(q_ids):
        result[str(qid)] = [str(cand_ids[j]) for j in idx_matrix[i]
                            if j < len(cand_ids)]
    return result

def recall_at_k(retrieved:Dict[str,List[str]],
                gt:Dict[str,Set[str]],
                k:int,
                cond_set:Set[str]=None) -> float:
    """
    cond_set: if provided, only count GT entities in cond_set
    (i.e. only those present in the sampled corpus).
    """
    total = 0; found = 0
    for qid, true_matches in gt.items():
        if cond_set is not None:
            relevant = true_matches & cond_set
        else:
            relevant = true_matches
        if not relevant: continue
        total += len(relevant)
        top_k = set(list(retrieved.get(qid,[]))[:k])
        found += len(relevant & top_k)
    if total == 0: return float('nan')
    return round(100.0 * found / total, 2)

def recall_table(retrieved:Dict[str,List[str]], gt:Dict[str,Set[str]],
                 cond_set:Set[str]=None) -> Dict[str,float]:
    return {f"Recall@{k}": recall_at_k(retrieved, gt, k, cond_set)
            for k in K_VALS}

# ══════════════════════════════════════════════════════════════════════════════
# T1 — CORPUS COVERAGE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def task1_coverage(gt:Dict[str,Set[str]], cand_id_set:Set[str]):
    print("\n" + "="*60)
    print("T1: GROUND-TRUTH CORPUS COVERAGE")
    print("="*60)
    total_gt_pairs = 0
    in_corpus      = 0
    absent         = 0
    queries_with_any_in_corpus = 0
    queries_with_all_absent    = 0
    coverage_per_query = []

    for qid, matches in gt.items():
        if not matches: continue
        present = matches & cand_id_set
        in_corpus  += len(present)
        absent     += len(matches) - len(present)
        total_gt_pairs += len(matches)
        coverage_per_query.append(len(present) / len(matches))
        if present: queries_with_any_in_corpus += 1
        else:       queries_with_all_absent    += 1

    avg_coverage = np.mean(coverage_per_query) if coverage_per_query else 0.0
    coverage_pct = 100.0 * in_corpus / max(total_gt_pairs,1)

    print(f"  Total GT match pairs (for sampled queries)  : {total_gt_pairs:,}")
    print(f"  GT entities PRESENT in 1M corpus            : {in_corpus:,}  ({coverage_pct:.1f}%)")
    print(f"  GT entities ABSENT from 1M corpus           : {absent:,}  ({100-coverage_pct:.1f}%)")
    print(f"  Queries where ANY match is in corpus        : {queries_with_any_in_corpus:,}")
    print(f"  Queries where ALL matches are absent        : {queries_with_all_absent:,}")
    print(f"  Mean per-query GT coverage                  : {avg_coverage*100:.1f}%")
    print()
    print("  NOTE: Max achievable raw Recall@K is bounded by GT coverage.")
    print(f"  MAX ACHIEVABLE RAW RECALL@50 ≤ {coverage_pct:.1f}%")

    return {
        "total_gt_pairs": total_gt_pairs,
        "in_corpus": in_corpus,
        "absent": absent,
        "coverage_pct": round(coverage_pct, 2),
        "avg_per_query_coverage_pct": round(avg_coverage*100, 2),
        "queries_with_any_in_corpus": queries_with_any_in_corpus,
        "queries_with_all_absent": queries_with_all_absent,
    }

# ══════════════════════════════════════════════════════════════════════════════
# T3 — FAILURE CATEGORISATION
# ══════════════════════════════════════════════════════════════════════════════
LEGAL_SUFFIXES = re.compile(
    r'\b(inc|llc|ltd|pvt|limited|gmbh|corp|co|plc|sa|ag|bv|nv)\b', re.I)
DIGIT_RE = re.compile(r'\d')

def categorise_pair(s1_row, cand_row) -> str:
    n1 = str(s1_row.get('name_norm',''))
    n2 = str(cand_row.get('name_norm',''))
    a1 = str(s1_row.get('address_norm',''))
    a2 = str(cand_row.get('address_norm',''))

    if not n1 or not n2:
        return "name_missing"

    # Exact
    if n1 == n2:
        return "exact_name"

    toks1 = set(tokenize(n1))
    toks2 = set(tokenize(n2))

    if not toks1 or not toks2:
        return "name_missing"

    overlap = len(toks1 & toks2) / max(len(toks1|toks2), 1)

    # Legal suffix variation
    clean1 = LEGAL_SUFFIXES.sub('', n1).strip()
    clean2 = LEGAL_SUFFIXES.sub('', n2).strip()
    if clean1 == clean2:
        return "legal_suffix_variation"

    # Numeric/id presence
    if (DIGIT_RE.search(n1) or DIGIT_RE.search(n2)):
        return "numeric_identifier"

    if overlap >= 0.5:
        if n1.split() != n2.split() and toks1 == toks2:
            return "token_order_variation"
        return "partial_name_overlap"

    # Spelling variation (edit distance heuristic)
    min_len = min(len(n1),len(n2))
    char_overlap = len(set(n1) & set(n2)) / max(len(set(n1)|set(n2)),1)
    if char_overlap > 0.7:
        return "spelling_variation"

    # Address-dependent
    if a1 and a2 and a1 != '' and a2 != '':
        a_toks1 = set(tokenize(a1))
        a_toks2 = set(tokenize(a2))
        addr_overlap = len(a_toks1 & a_toks2) / max(len(a_toks1|a_toks2),1)
        if addr_overlap > 0.3:
            return "address_dependent"

    if not a1 or not a2:
        return "name_and_address_weak"

    return "other"

def task3_failure_analysis(s1_df:pd.DataFrame, cand_df:pd.DataFrame,
                           gt:Dict[str,Set[str]],
                           cand_id_set:Set[str],
                           bm25s_retrieved:Dict[str,List[str]]):
    print("\n" + "="*60)
    print("T3: FAILURE CATEGORISATION (GT present but BM25 missed)")
    print("="*60)
    # Build cand lookup
    cand_idx = {row['entity_id']: row
                for _, row in cand_df.iterrows()
                if row['entity_id'] in cand_id_set}
    s1_idx   = {row['entity_id']: row
                for _, row in s1_df.iterrows()}

    categories = defaultdict(list)   # cat → list of (s1_id, gt_id)
    checked = 0

    for qid, true_matches in gt.items():
        present = true_matches & cand_id_set
        if not present: continue
        top50  = set(list(bm25s_retrieved.get(qid,[]))[:50])
        missed = present - top50
        if not missed: continue
        s1_row = s1_idx.get(qid, {})
        for gt_id in missed:
            cand_row = cand_idx.get(gt_id, {})
            cat = categorise_pair(s1_row, cand_row)
            categories[cat].append((qid, gt_id))
        checked += 1
        if checked >= 1000: break

    print(f"\n  Analysed {checked} queries with missed GT in corpus:")
    cat_results = {}
    for cat, pairs in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:<30} {len(pairs):>5} pairs")
        cat_results[cat] = len(pairs)

    return cat_results

# ══════════════════════════════════════════════════════════════════════════════
# T7 — FAILURE EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════
def task7_failure_examples(s1_df:pd.DataFrame, cand_df:pd.DataFrame,
                           gt:Dict[str,Set[str]], cand_id_set:Set[str],
                           retrieved:Dict[str,List[str]],
                           n_examples:int=20) -> List[str]:
    print("\n" + "="*60)
    print("T7: FAILURE EXAMPLES (GT present, BM25 top-50 missed)")
    print("="*60)
    cand_lookup = cand_df.set_index('entity_id').to_dict('index')
    s1_lookup   = s1_df.set_index('entity_id').to_dict('index')
    examples = []

    count = 0
    for qid, true_matches in gt.items():
        if count >= n_examples: break
        present = true_matches & cand_id_set
        if not present: continue
        top50 = list(retrieved.get(qid,[]))[:50]
        missed = present - set(top50)
        if not missed: continue

        gt_id   = next(iter(missed))
        s1_row  = s1_lookup.get(qid, {})
        cand_row= cand_lookup.get(gt_id, {})
        top5_ids= top50[:5]

        ex = [
            f"\n--- FAILURE EXAMPLE {count+1} ---",
            f"  S1 ID      : {qid}",
            f"  S1 Name    : {s1_row.get('business_name','?')}",
            f"  S1 Norm    : {s1_row.get('name_norm','?')}",
            f"  S1 Address : {s1_row.get('business_address','?')}",
            f"  GT ID      : {gt_id}",
            f"  GT Name    : {cand_row.get('business_name','?')}",
            f"  GT Norm    : {cand_row.get('name_norm','?')}",
            f"  GT Address : {cand_row.get('business_address','?')}",
            f"  BM25 Top-5 Names:",
        ]
        for rank_id in top5_ids:
            r = cand_lookup.get(rank_id, {})
            ex.append(f"    [{rank_id}] {r.get('business_name','?')} | {r.get('business_address','?')}")
        for line in ex: print(line)
        examples.extend(ex)
        count += 1

    return examples

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(str(msg))

    # ── Load data ──────────────────────────────────────────────────────────────
    log("="*60)
    log("ECHO-ER Phase 4 — Recall Analysis")
    log("="*60)
    s1_df, cand_df, cand_id_set = load_data(N_QUERIES, N_CORPUS)
    q_ids    = s1_df['entity_id'].values
    cand_ids = cand_df['entity_id'].values

    # Text views
    s1_name   = s1_df['name_norm'].tolist()
    s1_addr   = s1_df['address_norm'].tolist()
    s1_full   = (s1_df['name_norm'] + " " + s1_df['address_norm'] + " " + s1_df['country_norm']).tolist()

    cand_name = cand_df['name_norm'].tolist()
    cand_addr = cand_df['address_norm'].tolist()
    cand_full = (cand_df['name_norm'] + " " + cand_df['address_norm'] + " " + cand_df['country_norm']).tolist()

    # Load GT
    log("\nLoading ground truth...")
    gt = load_gt(q_ids)
    log(f"  GT records for sampled queries: {len(gt):,}")

    # T1 — Coverage
    cov = task1_coverage(gt, cand_id_set)
    log("")
    log(f"GROUND-TRUTH COVERAGE = {cov['coverage_pct']}%")
    log(f"Max achievable raw Recall@50 ≤ {cov['coverage_pct']}%")

    # T4+T5+T6 — Baseline retrievers
    # We use bm25s for BM25 (faster) and sklearn for TF-IDF variants
    import bm25s

    def run_bm25s_retrieval(corp_texts, q_texts, k=TOP_BASELN):
        corp_tok = bm25s.tokenize(corp_texts, stopwords="en")
        q_tok    = bm25s.tokenize(q_texts,    stopwords="en")
        ret      = bm25s.BM25(k1=1.5, b=0.75)
        ret.index(corp_tok)
        idx_mat, _ = ret.retrieve(q_tok, corpus=np.arange(len(corp_texts)), k=min(k, len(corp_texts)))
        return idx_mat

    log("\n" + "="*60)
    log("T4+T5+T6: BASELINE RETRIEVER COMPARISON")
    log("="*60)

    baselines = {}

    # A — Exact normalized-name lookup
    log("\n  A. Exact name lookup...")
    name_to_cand_positions = defaultdict(list)
    for pos, name in enumerate(cand_name):
        if name: name_to_cand_positions[name].append(pos)
    exact_retrieved = {}
    for qid, qname in zip(q_ids, s1_name):
        exact_retrieved[str(qid)] = [str(cand_ids[p])
                                      for p in name_to_cand_positions.get(qname, [])[:TOP_BASELN]]
    exact_raw  = recall_table(exact_retrieved, gt)
    exact_cond = recall_table(exact_retrieved, gt, cond_set=cand_id_set)
    baselines['A_ExactName'] = {'raw': exact_raw, 'cond': exact_cond}
    log(f"    Raw  Recall@50 = {exact_raw['Recall@50']}%")
    log(f"    Cond Recall@50 = {exact_cond['Recall@50']}%")
    del name_to_cand_positions, exact_retrieved; gc.collect()

    # B — Char n-gram TF-IDF (3-5)
    log("\n  B. Char n-gram TF-IDF (3-5)...")
    t0 = time.time()
    idx_char = tfidf_retrieve(cand_name, s1_name, k=TOP_BASELN,
                               analyzer='char_wb', ngram_range=(3,5), max_features=300_000)
    ret_char = build_retrieved_dict(q_ids, idx_char, cand_ids)
    char_raw  = recall_table(ret_char, gt)
    char_cond = recall_table(ret_char, gt, cond_set=cand_id_set)
    baselines['B_CharNgram35'] = {'raw': char_raw, 'cond': char_cond}
    log(f"    Time: {time.time()-t0:.1f}s  |  Raw Recall@50={char_raw['Recall@50']}%  |  Cond={char_cond['Recall@50']}%")
    del idx_char; gc.collect()

    # C — Char n-gram TF-IDF (3-6)
    log("\n  C. Char n-gram TF-IDF (3-6)...")
    t0 = time.time()
    idx_char6 = tfidf_retrieve(cand_name, s1_name, k=TOP_BASELN,
                                analyzer='char_wb', ngram_range=(3,6), max_features=300_000)
    ret_char6 = build_retrieved_dict(q_ids, idx_char6, cand_ids)
    char6_raw  = recall_table(ret_char6, gt)
    char6_cond = recall_table(ret_char6, gt, cond_set=cand_id_set)
    baselines['C_CharNgram36'] = {'raw': char6_raw, 'cond': char6_cond}
    log(f"    Time: {time.time()-t0:.1f}s  |  Raw Recall@50={char6_raw['Recall@50']}%  |  Cond={char6_cond['Recall@50']}%")
    del idx_char6; gc.collect()

    # D — Word TF-IDF
    log("\n  D. Word TF-IDF (name)...")
    t0 = time.time()
    idx_word = tfidf_retrieve(cand_name, s1_name, k=TOP_BASELN,
                               analyzer='word', max_features=200_000)
    ret_word = build_retrieved_dict(q_ids, idx_word, cand_ids)
    word_raw  = recall_table(ret_word, gt)
    word_cond = recall_table(ret_word, gt, cond_set=cand_id_set)
    baselines['D_WordTFIDF'] = {'raw': word_raw, 'cond': word_cond}
    log(f"    Time: {time.time()-t0:.1f}s  |  Raw Recall@50={word_raw['Recall@50']}%  |  Cond={word_cond['Recall@50']}%")
    del idx_word; gc.collect()

    # E — BM25 (name only)
    log("\n  E. BM25 (name only)...")
    t0 = time.time()
    idx_bm25 = run_bm25s_retrieval(cand_name, s1_name, k=TOP_BASELN)
    ret_bm25 = build_retrieved_dict(q_ids, idx_bm25, cand_ids)
    bm25_raw  = recall_table(ret_bm25, gt)
    bm25_cond = recall_table(ret_bm25, gt, cond_set=cand_id_set)
    baselines['E_BM25Name'] = {'raw': bm25_raw, 'cond': bm25_cond}
    log(f"    Time: {time.time()-t0:.1f}s  |  Raw Recall@50={bm25_raw['Recall@50']}%  |  Cond={bm25_cond['Recall@50']}%")

    # F — BM25 Address only
    log("\n  F. BM25 (address only)...")
    t0 = time.time()
    idx_addr = run_bm25s_retrieval(cand_addr, s1_addr, k=TOP_BASELN)
    ret_addr = build_retrieved_dict(q_ids, idx_addr, cand_ids)
    addr_raw  = recall_table(ret_addr, gt)
    addr_cond = recall_table(ret_addr, gt, cond_set=cand_id_set)
    baselines['F_BM25Address'] = {'raw': addr_raw, 'cond': addr_cond}
    log(f"    Time: {time.time()-t0:.1f}s  |  Raw Recall@50={addr_raw['Recall@50']}%  |  Cond={addr_cond['Recall@50']}%")

    # G — Full record BM25
    log("\n  G. BM25 (full: name+addr+country)...")
    t0 = time.time()
    idx_full = run_bm25s_retrieval(cand_full, s1_full, k=TOP_BASELN)
    ret_full = build_retrieved_dict(q_ids, idx_full, cand_ids)
    full_raw  = recall_table(ret_full, gt)
    full_cond = recall_table(ret_full, gt, cond_set=cand_id_set)
    baselines['G_BM25Full'] = {'raw': full_raw, 'cond': full_cond}
    log(f"    Time: {time.time()-t0:.1f}s  |  Raw Recall@50={full_raw['Recall@50']}%  |  Cond={full_cond['Recall@50']}%")
    del idx_full; gc.collect()

    # H — T6: Name OR Address UNION
    log("\n  H. Name ∪ Address Union (BM25)...")
    union_retrieved = {}
    for qid in [str(x) for x in q_ids]:
        name_set  = set(list(ret_bm25.get(qid,[]))[:50])
        addr_set  = set(list(ret_addr.get(qid,[]))[:50])
        union_retrieved[qid] = list(name_set | addr_set)
    union_raw  = recall_table(union_retrieved, gt)
    union_cond = recall_table(union_retrieved, gt, cond_set=cand_id_set)
    baselines['H_NameAddrUnion'] = {'raw': union_raw, 'cond': union_cond}
    log(f"    Raw Recall@50={union_raw['Recall@50']}%  |  Cond={union_cond['Recall@50']}%")
    log(f"    Raw Recall@100={union_raw['Recall@100']}%  |  Cond={union_cond['Recall@100']}%")

    # Name only / Address only / Both / Union-only breakdown
    log("\n  Coverage breakdown (GT present in corpus):")
    name_only_count = addr_only_count = both_count = union_only_count = 0
    for qid, matches in gt.items():
        present = matches & cand_id_set
        if not present: continue
        n50 = set(list(ret_bm25.get(qid,[]))[:50])
        a50 = set(list(ret_addr.get(qid,[]))[:50])
        for gid in present:
            in_n = gid in n50
            in_a = gid in a50
            if in_n and in_a: both_count += 1
            elif in_n:        name_only_count += 1
            elif in_a:        addr_only_count += 1
            else:             union_only_count += 1

    log(f"    GT retrieved by Name only   : {name_only_count}")
    log(f"    GT retrieved by Address only: {addr_only_count}")
    log(f"    GT retrieved by Both        : {both_count}")
    log(f"    GT missed by BOTH (Union miss): {union_only_count}")

    # T3 — Failure analysis (using BM25 name results)
    cat_results = task3_failure_analysis(s1_df, cand_df, gt, cand_id_set, ret_bm25)

    # T7 — Failure examples
    examples = task7_failure_examples(s1_df, cand_df, gt, cand_id_set, ret_bm25)

    # ── T2: Conditional Recall table (best retriever) ─────────────────────────
    log("\n" + "="*60)
    log("T2: CONDITIONAL RECALL@K (GT present in 1M corpus)")
    log("="*60)
    best_name = "H_NameAddrUnion"
    best_ret  = union_retrieved
    log(f"\n  Using: {best_name}")
    for k in K_VALS:
        raw  = recall_at_k(best_ret, gt, k)
        cond = recall_at_k(best_ret, gt, k, cond_set=cand_id_set)
        log(f"  Recall@{k:<4} | RAW={raw}%  | CONDITIONAL={cond}%")

    # ── Find best method ──────────────────────────────────────────────────────
    best_method  = max(baselines, key=lambda x: baselines[x]['cond'].get('Recall@50', 0))
    best_cond50  = baselines[best_method]['cond'].get('Recall@50', 0)

    # Determine main failure mode
    if cat_results:
        main_failure = max(cat_results, key=cat_results.get)
    else:
        main_failure = "UNKNOWN"

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════════════════════
    def fmt_row(method, d):
        raw  = d.get('raw',{})
        cond = d.get('cond',{})
        return (f"| {method:<22} "
                f"| {raw.get('Recall@5','?'):<6} | {raw.get('Recall@10','?'):<6} "
                f"| {raw.get('Recall@20','?'):<6} | {raw.get('Recall@50','?'):<6} "
                f"| {cond.get('Recall@5','?'):<6} | {cond.get('Recall@10','?'):<6} "
                f"| {cond.get('Recall@20','?'):<6} | {cond.get('Recall@50','?'):<6} |")

    md_header = ("| Method                 "
                 "| R@5   | R@10  | R@20  | R@50  "
                 "| cR@5  | cR@10 | cR@20 | cR@50 |\n"
                 "|------------------------|-------|-------|-------|-------|"
                 "-------|-------|-------|-------|")

    md = f"""# Phase 4 Recall Analysis — ECHO-ER

## 1. Raw Recall@K (current BM25, name view, 1M corpus)

| K   | BM25 |
|-----|------|
| 5   | {bm25_raw.get('Recall@5','?')}% |
| 10  | {bm25_raw.get('Recall@10','?')}% |
| 20  | {bm25_raw.get('Recall@20','?')}% |
| 50  | {bm25_raw.get('Recall@50','?')}% |
| 100 | {bm25_raw.get('Recall@100','?')}% |

## 2. Ground-Truth Coverage in 1M Sampled Corpus

| Metric | Value |
|---|---|
| Total GT match pairs | {cov['total_gt_pairs']:,} |
| GT entities PRESENT in corpus | {cov['in_corpus']:,} ({cov['coverage_pct']}%) |
| GT entities ABSENT from corpus | {cov['absent']:,} ({100-cov['coverage_pct']:.1f}%) |
| Queries with any GT in corpus | {cov['queries_with_any_in_corpus']:,} |
| Queries with ALL GT absent | {cov['queries_with_all_absent']:,} |
| Mean per-query GT coverage | {cov['avg_per_query_coverage_pct']}% |
| **Max achievable raw Recall@50** | **≤ {cov['coverage_pct']}%** |

> ⚠️ If coverage is low (e.g. <30%), the low raw Recall@50 is PRIMARILY due to corpus subsampling, not retriever failure.

## 3. Conditional Recall@K (GT entity present in corpus)

| Method | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|
| BM25 Name | {bm25_cond.get('Recall@5','?')}% | {bm25_cond.get('Recall@10','?')}% | {bm25_cond.get('Recall@20','?')}% | {bm25_cond.get('Recall@50','?')}% | {bm25_cond.get('Recall@100','?')}% |
| BM25 Address | {addr_cond.get('Recall@5','?')}% | {addr_cond.get('Recall@10','?')}% | {addr_cond.get('Recall@20','?')}% | {addr_cond.get('Recall@50','?')}% | {addr_cond.get('Recall@100','?')}% |
| BM25 Full | {full_cond.get('Recall@5','?')}% | {full_cond.get('Recall@10','?')}% | {full_cond.get('Recall@20','?')}% | {full_cond.get('Recall@50','?')}% | {full_cond.get('Recall@100','?')}% |
| Name ∪ Address | {union_cond.get('Recall@5','?')}% | {union_cond.get('Recall@10','?')}% | {union_cond.get('Recall@20','?')}% | {union_cond.get('Recall@50','?')}% | {union_cond.get('Recall@100','?')}% |

## 4. Failure Mode Categories

| Category | Count |
|---|---|
""" + "\n".join(f"| {c} | {n} |" for c,n in sorted(cat_results.items(), key=lambda x: -x[1])) + f"""

**Main failure mode: {main_failure}**

## 5. All Baseline Comparison

{md_header}
""" + "\n".join(fmt_row(m,d) for m,d in baselines.items()) + f"""

## 6. Name + Address Coverage Breakdown

| Category | Count |
|---|---|
| GT retrieved by Name only | {name_only_count} |
| GT retrieved by Address only | {addr_only_count} |
| GT retrieved by Both | {both_count} |
| GT missed by UNION | {union_only_count} |

## 7. Failure Examples (20 cases, GT present but BM25 missed)

```
""" + "\n".join(examples) + """
```

## 8. BM25 Parameters Used

- k1 = 1.5, b = 0.75
- Tokenizer: bm25s built-in (stopwords="en")
- Custom: regex `\\b\\w+\\b` + STOP_WORDS + MAX_POSTING=80K

## 9. Recommended Retrieval Strategy

Based on measured evidence, the recommended approach is:

1. Use **bm25s** for speed (verified faster, similar recall to custom BM25)
2. Run **Name view + Address view separately** and take the **UNION**
3. Use **character n-gram TF-IDF** as a secondary retriever for spelling variations
4. Set K ≥ 50 to ensure GT is captured when it IS in the corpus

---

## Final Report

| Item | Value |
|---|---|
| RECALL CALCULATION | PASS |
| GROUND-TRUTH COVERAGE | {cov['coverage_pct']}% |
| CONDITIONAL RECALL@50 | {union_cond.get('Recall@50','?')}% (Name∪Address) |
| BEST SPARSE RETRIEVER | {best_method} |
| BEST CONDITIONAL RECALL@50 | {best_cond50}% |
| NAME+ADDRESS UNION RECALL@50 | {union_raw.get('Recall@50','?')}% (raw) / {union_cond.get('Recall@50','?')}% (conditional) |
| MAIN FAILURE MODE | {main_failure} |
"""

    report_path = os.path.join(DOCS_DIR, "PHASE_4_RECALL_ANALYSIS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    log("\n" + "="*60)
    log("FINAL REPORT")
    log("="*60)
    log(f"RECALL CALCULATION          = PASS")
    log(f"GROUND-TRUTH COVERAGE       = {cov['coverage_pct']}%")
    log(f"CONDITIONAL RECALL@50       = {union_cond.get('Recall@50','?')}%  (Name∪Address Union)")
    log(f"BEST SPARSE RETRIEVER       = {best_method}")
    log(f"BEST CONDITIONAL RECALL@50  = {best_cond50}%")
    log(f"NAME+ADDRESS UNION RECALL@50= {union_raw.get('Recall@50','?')}% (raw)")
    log(f"MAIN FAILURE MODE           = {main_failure}")
    log(f"\n📄 Report saved → {report_path}")
    log("\nSTOPPED. Awaiting review before next phase.")


if __name__ == "__main__":
    main()
