import os
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import gc
import pickle

# ---------------------------------------------------------
# FAST SPARSE DOT-PRODUCT K-NN (C++ Backed via Scipy)
# ---------------------------------------------------------
def _process_chunk_cpu(args):
    start_idx, end_idx, q_chunk, corpus_matrix, top_k = args
    sim_matrix = q_chunk.dot(corpus_matrix.T)
    
    num_queries_chunk = q_chunk.shape[0]
    chunk_indices = np.zeros((num_queries_chunk, top_k), dtype=np.int32)
    chunk_scores = np.zeros((num_queries_chunk, top_k), dtype=np.float32)
    
    for i in range(num_queries_chunk):
        row = sim_matrix.getrow(i)
        nonzero_indices = row.indices
        nonzero_data = row.data
        
        if len(nonzero_data) == 0:
            continue
            
        if len(nonzero_data) > top_k:
            top_indices_unsorted = np.argpartition(nonzero_data, -top_k)[-top_k:]
            top_scores_unsorted = nonzero_data[top_indices_unsorted]
            
            sort_order = np.argsort(-top_scores_unsorted)
            best_indices = top_indices_unsorted[sort_order]
            best_scores = top_scores_unsorted[sort_order]
        else:
            sort_order = np.argsort(-nonzero_data)
            best_indices = np.arange(len(nonzero_data))[sort_order]
            best_scores = nonzero_data[sort_order]
            
        actual_indices = nonzero_indices[best_indices]
        num_found = min(top_k, len(best_scores))
        chunk_indices[i, :num_found] = actual_indices[:num_found]
        chunk_scores[i, :num_found] = best_scores[:num_found]
        
    return start_idx, chunk_indices, chunk_scores

def get_top_k_sparse(query_matrix, corpus_matrix, top_k=30, batch_size=20000):
    """
    Computes cosine similarity (dot product of L2 normalized TF-IDF).
    Attempts GPU (CuPy) if available. Falls back to Multi-CPU parallel chunks.
    """
    num_queries = query_matrix.shape[0]
    top_k_indices = np.zeros((num_queries, top_k), dtype=np.int32)
    top_k_scores = np.zeros((num_queries, top_k), dtype=np.float32)
    
    # 1. Attempt GPU Acceleration via CuPy
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sp
        print("🔥 GPU (CuPy) detected! Offloading sparse dot-product to GPU...")
        
        # Move corpus to GPU (once)
        corpus_gpu = cpx_sp.csr_matrix(corpus_matrix)
        
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_chunk_gpu = cpx_sp.csr_matrix(query_matrix[start_idx:end_idx])
            
            # GPU Matrix Multiplication
            sim_matrix_gpu = q_chunk_gpu.dot(corpus_gpu.T)
            
            # Move back to CPU for Top-K argsort (sorting sparse on GPU is tricky without dense cast)
            sim_matrix_cpu = sim_matrix_gpu.get()
            
            # CPU Fallback for sorting the chunk
            _, c_idx, c_scores = _process_chunk_cpu((start_idx, end_idx, sim_matrix_cpu, sp.eye(corpus_matrix.shape[0]), top_k)) 
            # Note: since sim_matrix is already computed, we pass an identity trick, or better yet, just extract rows.
            # Actually, to keep it perfectly clean, let's just do CPU parallel if GPU isn't perfectly seamless for sparse argpartition.
            raise ImportError("Fallback to optimized CPU parallelization.")
            
    except ImportError:
        print("⚡ Using Multi-CPU Parallel Sparse Dot-Product...")
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        num_cores = multiprocessing.cpu_count()
        chunk_args = []
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            q_chunk = query_matrix[start_idx:end_idx]
            chunk_args.append((start_idx, end_idx, q_chunk, corpus_matrix, top_k))
            
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for start_idx, chunk_idx, chunk_scores in executor.map(_process_chunk_cpu, chunk_args):
                end_idx = start_idx + chunk_idx.shape[0]
                top_k_indices[start_idx:end_idx, :] = chunk_idx
                top_k_scores[start_idx:end_idx, :] = chunk_scores
                
    return top_k_indices, top_k_scores

def build_view_string(row, view_name):
    if view_name == 'Name':
        return str(row.get('name_norm', ''))
    elif view_name == 'Address':
        return str(row.get('address_norm', '')) + " " + str(row.get('postal_code', ''))
    else:
        return str(row.get('name_norm', '')) + " " + str(row.get('address_norm', '')) + " " + str(row.get('country_norm', ''))

# ---------------------------------------------------------
# PHASE 3 & 4 (CACHE & CHECKPOINT ENABLED)
# ---------------------------------------------------------
def run_kaggle_fast_retrieval():
    processed_dir = "/kaggle/working/data/processed/train"
    out_dir = "/kaggle/working/data/retrieval/train"
    cache_dir = "/kaggle/working/data/cache"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    VIEWS = ['Name', 'Address', 'Full']
    TOP_K = 30
    
    print("⏳ Loading IDs...")
    s1_df = pd.read_csv(os.path.join(processed_dir, "train_source1_norm.tsv"), sep='\t', usecols=['entity_id'])
    s1_ids = s1_df['entity_id'].values
    
    s2_df = pd.read_csv(os.path.join(processed_dir, "train_source2_norm.tsv"), sep='\t', usecols=['entity_id'])
    s3_df = pd.read_csv(os.path.join(processed_dir, "train_source3_norm.tsv"), sep='\t', usecols=['entity_id'])
    candidate_ids = np.concatenate([s2_df['entity_id'].values, s3_df['entity_id'].values])
    
    del s1_df, s2_df, s3_df; gc.collect()
    
    # Process ONE VIEW AT A TIME to save RAM and checkpoint
    for view in VIEWS:
        checkpoint_path = os.path.join(out_dir, f"tfidf_{view.lower()}_top{TOP_K}.parquet")
        
        if os.path.exists(checkpoint_path):
            print(f"✅ Checkpoint found for {view} View. Skipping to next...")
            continue
            
        print(f"\n🚀 --- Processing {view} View ---")
        t0 = time.time()
        
        # Load just the columns needed for this view
        cols_to_load = ['entity_id']
        if view == 'Name': cols_to_load += ['name_norm']
        elif view == 'Address': cols_to_load += ['address_norm', 'postal_code']
        else: cols_to_load += ['name_norm', 'address_norm', 'country_norm', 'postal_code']
        
        s1 = pd.read_csv(os.path.join(processed_dir, "train_source1_norm.tsv"), sep='\t', usecols=cols_to_load).fillna('')
        s2 = pd.read_csv(os.path.join(processed_dir, "train_source2_norm.tsv"), sep='\t', usecols=cols_to_load).fillna('')
        s3 = pd.read_csv(os.path.join(processed_dir, "train_source3_norm.tsv"), sep='\t', usecols=cols_to_load).fillna('')
        
        print("Creating view strings...")
        s1_texts = s1.apply(lambda r: build_view_string(r, view), axis=1).tolist()
        cand_texts = s2.apply(lambda r: build_view_string(r, view), axis=1).tolist() + s3.apply(lambda r: build_view_string(r, view), axis=1).tolist()
        
        del s1, s2, s3; gc.collect()
        
        print(f"Vectorizing (TF-IDF)...")
        # Use single characters and words
        vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=500000)
        
        # Fit and transform candidates
        cand_matrix = vectorizer.fit_transform(cand_texts)
        # Transform queries
        q_matrix = vectorizer.transform(s1_texts)
        
        del cand_texts, s1_texts; gc.collect()
        
        print(f"✅ Vectorization complete. Query matrix: {q_matrix.shape}, Corpus matrix: {cand_matrix.shape}")
        
        print(f"🔍 Searching Top-{TOP_K} via Sparse C++ BLAS (Chunked)...")
        results_idx, results_scores = get_top_k_sparse(q_matrix, cand_matrix, top_k=TOP_K, batch_size=5000)
        
        del q_matrix, cand_matrix, vectorizer; gc.collect()
        
        print(f"💾 Saving {view} View Checkpoint...")
        
        # Flatten and save
        n_queries = len(s1_ids)
        q_ids_rep = np.repeat(s1_ids, TOP_K)
        c_ids_flat = candidate_ids[results_idx.flatten()]
        scores_flat = results_scores.flatten()
        ranks_flat = np.tile(np.arange(1, TOP_K + 1), n_queries)
        
        view_df = pd.DataFrame({
            's1_id': q_ids_rep,
            'candidate_id': c_ids_flat,
            'view': view,
            'rank': ranks_flat,
            'score': scores_flat
        })
        
        # Filter out 0.0 scores (no overlap)
        view_df = view_df[view_df['score'] > 0]
        
        view_df.to_parquet(checkpoint_path, index=False)
        print(f"✅ {view} View complete in {time.time() - t0:.2f}s.")
        
    print("\n🎉 All views processed and checkpointed!")

if __name__ == "__main__":
    run_kaggle_fast_retrieval()
