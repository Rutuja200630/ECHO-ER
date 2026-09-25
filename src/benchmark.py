import os
import time
import tracemalloc
import pandas as pd
from retrieval import SparseRetriever, batch_retrieve, create_views

def benchmark(n_queries=10000, n_corpus=100000):
    print(f"=== BENCHMARK: Queries={n_queries}, Corpus={n_corpus} ===")
    data_dir = r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\train"
    
    # We will use the raw TSVs and just apply a quick normalization view 
    print("Loading datasets...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str, nrows=n_queries)
    
    # We need n_corpus from S2/S3
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str, nrows=n_corpus)
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str, nrows=n_corpus)
    
    df_corpus = pd.concat([df_s2, df_s3], ignore_index=True).head(n_corpus)
    
    print("Creating views...")
    df_s1['norm_name'] = df_s1['business_name'].fillna("").str.lower()
    df_s1['norm_address'] = df_s1['business_address'].fillna("").str.lower()
    df_s1['norm_country'] = df_s1['country'].fillna("").str.lower()
    
    df_corpus['norm_name'] = df_corpus['business_name'].fillna("").str.lower()
    df_corpus['norm_address'] = df_corpus['business_address'].fillna("").str.lower()
    df_corpus['norm_country'] = df_corpus['country'].fillna("").str.lower()
    
    df_s1 = create_views(df_s1)
    df_corpus = create_views(df_corpus)
    
    # Start tracemalloc
    tracemalloc.start()
    
    t0 = time.time()
    print("Building TF-IDF Retriever for Name View...")
    retriever_name = SparseRetriever(method='tfidf')
    retriever_name.fit(df_corpus, id_col='entity_id', text_col='name_view')
    t_fit = time.time() - t0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    print(f"Fit Time: {t_fit:.2f}s | Peak RAM: {peak_mem / 1024 / 1024:.2f} MB")
    
    t1 = time.time()
    print("Retrieving candidates based on Name View...")
    res_name = batch_retrieve(df_s1, retriever_name, 'entity_id', 'name_view', 'name_view', k=20)
    t_ret = time.time() - t1
    
    print(f"Retrieval Time: {t_ret:.2f}s | Candidates Generated: {len(res_name) if res_name is not None else 0}")
    
    tracemalloc.stop()
    
    # Calculate Recall
    print("Calculating Recall@K...")
    gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
    
    # Filter GT to our queries
    gt = gt[gt['source1_entity_id'].isin(df_s1['entity_id'])]
    gt_map = {}
    for _, row in gt.iterrows():
        matches = row['matched_entity_ids']
        if pd.isna(matches):
            gt_map[row['source1_entity_id']] = set()
        else:
            gt_map[row['source1_entity_id']] = set(matches.split(','))
            
    # Compute recall
    if res_name is not None and not res_name.empty:
        # Group by S1 id, ordered by rank
        res_name_sorted = res_name.sort_values(['s1_id', 'rank'])
        
        recall_at_k = {5: 0, 10: 0, 20: 0, 50: 0}
        total_queries_with_gt = 0
        total_valid_retrieved = 0
        
        for q_id, group in res_name_sorted.groupby('s1_id'):
            true_matches = gt_map.get(q_id, set())
            if not true_matches:
                continue
                
            total_queries_with_gt += 1
            retrieved_ids = group['s2_id'].tolist()
            
            # Check correctness criteria
            # - no accidental self-matches (s2_id != s1_id)
            for c_id in retrieved_ids:
                if c_id == q_id:
                    print(f"WARNING: Self-match found for {q_id}")
            
            for k in recall_at_k.keys():
                top_k = set(retrieved_ids[:k])
                # Recall is true positives / total true positives for that query
                hits = len(top_k.intersection(true_matches))
                # Add to aggregate recall
                recall_at_k[k] += hits / len(true_matches)
                
        if total_queries_with_gt > 0:
            for k in recall_at_k.keys():
                avg_recall = recall_at_k[k] / total_queries_with_gt
                print(f"Recall@{k}: {avg_recall:.4f}")
    else:
        print("No candidates retrieved!")
        
    print("Done.")

if __name__ == "__main__":
    benchmark(10000, 100000)
