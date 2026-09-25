import os
import time
import pandas as pd
import numpy as np
import subprocess
import sys
import multiprocessing

# ---------------------------------------------------------
# ENSURE BM25S IS INSTALLED (Ultra-fast BM25 for Python)
# ---------------------------------------------------------
try:
    import bm25s
except ImportError:
    print("🚀 Installing bm25s for blazing-fast sparse retrieval...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bm25s"])
    import bm25s

# ---------------------------------------------------------
# PHASE 3: MULTI-VIEW REPRESENTATION BUILDER
# ---------------------------------------------------------
def create_views(df):
    """
    Creates the 3 retrieval views for the records.
    1. NAME VIEW: Just the normalized business name.
    2. ADDRESS VIEW: The normalized address + postal code.
    3. FULL VIEW: Name + Address + Country + Postal Code (The most context).
    """
    # Name View
    name_view = df['name_norm'].fillna("").astype(str)
    
    # Address View (Address + Postal Code)
    addr = df['address_norm'].fillna("").astype(str)
    postal = df['postal_code'].fillna("").astype(str)
    addr_view = addr + " " + postal
    
    # Full View
    country = df['country_norm'].fillna("").astype(str)
    full_view = name_view + " " + addr_view + " " + country
    
    return name_view.values, addr_view.values, full_view.values

# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------
def calculate_recall(retrieved_pairs, ground_truth_df, k_values=[5, 10, 20, 30, 50]):
    """
    Calculates Recall@K.
    retrieved_pairs: DataFrame with ['s1_id', 'candidate_id', 'rank']
    """
    # Create ground truth mapping: s1_id -> set of matched_entity_ids
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or matches == 'nan':
            gt_map[s1] = set()
        else:
            gt_map[s1] = set(matches.split(','))
            
    total_possible_matches = sum(len(v) for v in gt_map.values())
    if total_possible_matches == 0:
        return {k: 0 for k in k_values}
        
    recalls = {}
    
    for k in k_values:
        # Filter to top K
        top_k = retrieved_pairs[retrieved_pairs['rank'] <= k]
        
        # Group retrieved candidates by S1
        retrieved_dict = top_k.groupby('s1_id')['candidate_id'].apply(set).to_dict()
        
        matches_found = 0
        for s1, true_matches in gt_map.items():
            if not true_matches: continue
            retrieved_for_s1 = retrieved_dict.get(s1, set())
            matches_found += len(true_matches.intersection(retrieved_for_s1))
            
        recalls[f"Recall@{k}"] = round((matches_found / total_possible_matches) * 100, 2)
        
    return recalls

# ---------------------------------------------------------
# PHASE 4: SPARSE RETRIEVAL PIPELINE
# ---------------------------------------------------------
def run_kaggle_phase3_4():
    import glob
    
    processed_dir = "/kaggle/working/data/processed/train"
    out_dir = "/kaggle/working/data/retrieval/train"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    print("⏳ Loading normalized datasets into memory...")
    start_time = time.time()
    
    # Load S1 (Queries)
    s1_df = pd.read_csv(os.path.join(processed_dir, "train_source1_norm.tsv"), sep='\t')
    s1_ids = s1_df['entity_id'].values
    
    # Load S2 & S3 (Candidate Pool)
    s2_df = pd.read_csv(os.path.join(processed_dir, "train_source2_norm.tsv"), sep='\t')
    s3_df = pd.read_csv(os.path.join(processed_dir, "train_source3_norm.tsv"), sep='\t')
    candidates_df = pd.concat([s2_df, s3_df], ignore_index=True)
    candidate_ids = candidates_df['entity_id'].values
    
    print(f"✅ Loaded {len(s1_df):,} queries and {len(candidates_df):,} candidates in {time.time() - start_time:.2f}s.")
    
    # 2. Phase 3: Create Views
    print("⏳ Creating Multi-View representations...")
    s1_name, s1_addr, s1_full = create_views(s1_df)
    cand_name, cand_addr, cand_full = create_views(candidates_df)
    
    views = {
        "Name": (cand_name, s1_name),
        "Address": (cand_addr, s1_addr),
        "Full": (cand_full, s1_full)
    }
    
    del s1_df, s2_df, s3_df, candidates_df # Free RAM!
    
    # 3. Phase 4: BM25 Retrieval
    TOP_K = 30
    all_retrieval_results = []
    
    for view_name, (corpus_texts, query_texts) in views.items():
        print(f"\n🚀 --- Indexing {view_name} View ---")
        t0 = time.time()
        
        # Tokenize
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
        query_tokens = bm25s.tokenize(query_texts, stopwords="en")
        
        # Index
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        print(f"✅ Indexing complete in {time.time() - t0:.2f}s.")
        
        # Retrieve
        t1 = time.time()
        print(f"🔍 Searching Top-{TOP_K} for {len(query_texts):,} queries...")
        # Get indices and scores. bm25s uses multi-threading automatically.
        results_idx, results_scores = retriever.retrieve(query_tokens, corpus=candidate_ids, k=TOP_K)
        print(f"✅ Retrieval complete in {time.time() - t1:.2f}s.")
        
        # Flatten results into a DataFrame
        # results_idx has shape (num_queries, TOP_K)
        # results_scores has shape (num_queries, TOP_K)
        n_queries = len(s1_ids)
        
        # Construct arrays for the dataframe
        q_ids_rep = np.repeat(s1_ids, TOP_K)
        c_ids_flat = results_idx.flatten()
        scores_flat = results_scores.flatten()
        ranks_flat = np.tile(np.arange(1, TOP_K + 1), n_queries)
        
        view_df = pd.DataFrame({
            's1_id': q_ids_rep,
            'candidate_id': c_ids_flat,
            'view': view_name,
            'rank': ranks_flat,
            'score': scores_flat
        })
        
        # Save this view's results
        out_path = os.path.join(out_dir, f"bm25_{view_name.lower()}_top{TOP_K}.parquet")
        view_df.to_parquet(out_path, index=False)
        all_retrieval_results.append(view_df)
    
    # 4. Calculate Recall
    print("\n📊 Calculating Recall against Ground Truth...")
    gt_paths = glob.glob("/kaggle/input/**/train_ground_truth.tsv", recursive=True)
    if not gt_paths:
        print("⚠️ Could not find ground truth file to calculate recall.")
        return
        
    gt_df = pd.read_csv(gt_paths[0], sep='\t')
    
    # Evaluate each view separately
    for view_df in all_retrieval_results:
        view_name = view_df['view'].iloc[0]
        recalls = calculate_recall(view_df, gt_df, k_values=[5, 10, 20, 30])
        print(f"--> {view_name} View Recall: {recalls}")
        
    # Evaluate UNION of all views
    print("\n🔗 Evaluating Candidate Union (All Views Combined)...")
    combined_df = pd.concat(all_retrieval_results, ignore_index=True)
    # Get the best rank across any view for each candidate
    union_df = combined_df.groupby(['s1_id', 'candidate_id'])['rank'].min().reset_index()
    
    union_recalls = calculate_recall(union_df, gt_df, k_values=[5, 10, 20, 30])
    print(f"--> UNION Recall: {union_recalls}")
    print("\n🎉 Phase 3 and 4 Complete!")

if __name__ == "__main__":
    run_kaggle_phase3_4()
