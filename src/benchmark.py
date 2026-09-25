import os
import sys
import time
import tracemalloc
import pandas as pd
import argparse
import numpy as np
from retrieval import OldSparseRetriever, SparseRetriever, batch_retrieve, create_views

def compute_recall(res_df, gt_map, k_list=[5, 10, 20, 50, 100]):
    if res_df is None or res_df.empty:
        return {k: 0.0 for k in k_list}
        
    res_sorted = res_df.sort_values(['s1_id', 'rank'])
    recall_at_k = {k: 0 for k in k_list}
    total_queries = 0
    
    for q_id, group in res_sorted.groupby('s1_id'):
        true_matches = gt_map.get(q_id, set())
        if not true_matches:
            continue
        total_queries += 1
        retrieved_ids = group['s2_id'].tolist()
        
        for k in k_list:
            top_k = set(retrieved_ids[:k])
            hits = len(top_k.intersection(true_matches))
            recall_at_k[k] += hits / len(true_matches)
            
    if total_queries > 0:
        return {k: recall_at_k[k] / total_queries for k in k_list}
    return {k: 0.0 for k in k_list}

def run_retrieval(retriever_class, df_corpus, df_s1, name, k=100, **kwargs):
    print(f"\n--- Running {name} ---")
    tracemalloc.start()
    t0 = time.time()
    
    retriever = retriever_class(method='tfidf')
    retriever.fit(df_corpus, id_col='entity_id', text_col='name_view')
    t_fit = time.time() - t0
    
    t1 = time.time()
    res = batch_retrieve(df_s1, retriever, 'entity_id', 'name_view', 'name_view', k=k, **kwargs)
    t_ret = time.time() - t1
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    total_time = t_fit + t_ret
    peak_mb = peak_mem / 1024 / 1024
    
    print(f"Fit Time: {t_fit:.2f}s | Retrieval Time: {t_ret:.2f}s | Total Time: {total_time:.2f}s")
    print(f"Peak RAM: {peak_mb:.2f} MB | Candidates: {len(res)}")
    
    return res, {"fit_time": t_fit, "ret_time": t_ret, "total_time": total_time, "peak_mb": peak_mb}

def benchmark_correctness(data_dir):
    print("\n=== CORRECTNESS BENCHMARK (Queries=1000, Corpus=10000) ===")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str, nrows=1000)
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str, nrows=10000)
    df_corpus = pd.concat([df_s2], ignore_index=True)
    
    df_s1['norm_name'] = df_s1['business_name'].fillna("").str.lower()
    df_corpus['norm_name'] = df_corpus['business_name'].fillna("").str.lower()
    df_s1['norm_address'] = df_s1['business_address'].fillna("").str.lower()
    df_corpus['norm_address'] = df_corpus['business_address'].fillna("").str.lower()
    df_s1['norm_country'] = df_s1['country'].fillna("").str.lower()
    df_corpus['norm_country'] = df_corpus['country'].fillna("").str.lower()
    df_s1 = create_views(df_s1)
    df_corpus = create_views(df_corpus)
    
    # Use 100 for exact K
    from retrieval import SparseRetriever, InvertedIndexRetriever, OldSparseRetriever
    res_old, _ = run_retrieval(SparseRetriever, df_corpus, df_s1, "FULL SPARSE MATRIX", k=100, batch_size=5000, corpus_chunk_size=10000)
    res_new, _ = run_retrieval(InvertedIndexRetriever, df_corpus, df_s1, "INVERTED INDEX", k=100)
    
    print("\nComparing Results...")
    res_old_grp = res_old.groupby('s1_id')
    res_new_grp = res_new.groupby('s1_id')
    
    overlaps = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    count = 0
    
    for q_id in df_s1['entity_id']:
        if q_id not in res_old_grp.groups or q_id not in res_new_grp.groups:
            continue
            
        count += 1
        old_cands = res_old_grp.get_group(q_id).sort_values('rank')
        new_cands = res_new_grp.get_group(q_id).sort_values('rank')
        
        old_ids = old_cands['s2_id'].tolist()
        new_ids = new_cands['s2_id'].tolist()
        
        for k in overlaps.keys():
            overlaps[k] += len(set(old_ids[:k]).intersection(set(new_ids[:k]))) / k
                    
    print(f"Top-1 Agreement: {overlaps[1]/count:.4f}")
    print(f"Top-5 Overlap: {overlaps[5]/count:.4f}")
    print(f"Top-10 Overlap: {overlaps[10]/count:.4f}")
    print(f"Top-20 Overlap: {overlaps[20]/count:.4f}")
    print(f"Top-50 Overlap: {overlaps[50]/count:.4f}")
    print(f"Top-100 Overlap: {overlaps[100]/count:.4f}")

def benchmark_scaling(data_dir, n_queries, n_corpus, gt_map, chunk_size=500000):
    print(f"\n=== BENCHMARK (Queries={n_queries}, Corpus={n_corpus}) ===")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str, nrows=n_queries)
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str, nrows=n_corpus)
    df_corpus = pd.concat([df_s2], ignore_index=True)
    
    df_s1['norm_name'] = df_s1['business_name'].fillna("").str.lower()
    df_corpus['norm_name'] = df_corpus['business_name'].fillna("").str.lower()
    df_s1['norm_address'] = df_s1['business_address'].fillna("").str.lower()
    df_corpus['norm_address'] = df_corpus['business_address'].fillna("").str.lower()
    df_s1['norm_country'] = df_s1['country'].fillna("").str.lower()
    df_corpus['norm_country'] = df_corpus['country'].fillna("").str.lower()
    df_s1 = create_views(df_s1)
    df_corpus = create_views(df_corpus)
    
    from retrieval import SparseRetriever, InvertedIndexRetriever
    
    res_full, stats_full = run_retrieval(SparseRetriever, df_corpus, df_s1, "FULL SPARSE MATRIX", k=100, batch_size=5000, corpus_chunk_size=chunk_size)
    print("Calculating Recall@K (Full Sparse)...")
    recall_full = compute_recall(res_full, gt_map)
    for k, v in recall_full.items():
        print(f"Recall@{k}: {v:.4f}")
        
    res_inv, stats_inv = run_retrieval(InvertedIndexRetriever, df_corpus, df_s1, "INVERTED INDEX", k=100)
    print("Calculating Recall@K (Inverted Index)...")
    recall_inv = compute_recall(res_inv, gt_map)
    for k, v in recall_inv.items():
        print(f"Recall@{k}: {v:.4f}")
        
    print("\n=== SUMMARY ===")
    speedup = stats_full['ret_time'] / stats_inv['ret_time'] if stats_inv['ret_time'] > 0 else 0
    print(f"Retrieval Speedup: {speedup:.2f}x")
    print(f"Peak RAM Reduction: {stats_full['peak_mb'] - stats_inv['peak_mb']:.2f} MB")
    print(f"Recall@50 Diff: {recall_inv[50] - recall_full[50]:.4f}")
    print(f"Recall@100 Diff: {recall_inv[100] - recall_full[100]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()
    
    print("Loading Ground Truth...")
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
    gt_map = {}
    for _, row in gt.iterrows():
        matches = row['matched_entity_ids']
        if pd.isna(matches):
            gt_map[row['source1_entity_id']] = set()
        else:
            gt_map[row['source1_entity_id']] = set(matches.split(','))
            
    benchmark_correctness(args.data_dir)
    benchmark_scaling(args.data_dir, 10000, 100000, gt_map)
    benchmark_scaling(args.data_dir, 10000, 1000000, gt_map)
    print("\nBenchmarks Complete.")
