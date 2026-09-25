import numpy as np
import scipy.sparse as sp
import pandas as pd
import time

def mock_batch_retrieve():
    batch_size = 5
    k = 3
    num_corpus = 100
    
    # Mock scores matrix on CPU (from GPU .get())
    # 5 queries, 100 corpus. Each query has some non-zeros
    indptr = np.array([0, 2, 5, 5, 7, 10], dtype=np.int32)
    indices = np.array([10, 20, 10, 30, 40, 50, 60, 10, 50, 99], dtype=np.int32)
    data = np.array([0.5, 0.8, 0.2, 0.9, 0.1, 0.6, 0.7, 0.3, 0.4, 0.99], dtype=np.float32)
    
    scores_mat_cpu = sp.csr_matrix((data, indices, indptr), shape=(batch_size, num_corpus))
    
    global_top_scores = np.full((batch_size, k), -1.0, dtype=np.float32)
    global_top_indices = np.full((batch_size, k), -1, dtype=np.int32)
    
    # Python fallback
    for i in range(batch_size):
        row_start = scores_mat_cpu.indptr[i]
        row_end = scores_mat_cpu.indptr[i+1]
        if row_start == row_end:
            continue
        row_data = scores_mat_cpu.data[row_start:row_end]
        row_cols = scores_mat_cpu.indices[row_start:row_end]
        
        if len(row_data) > k:
            top_k_local = np.argpartition(row_data, -k)[-k:]
            global_top_scores[i] = row_data[top_k_local]
            global_top_indices[i] = row_cols[top_k_local]
        else:
            global_top_scores[i, :len(row_data)] = row_data
            global_top_indices[i, :len(row_cols)] = row_cols
            
    print("Global Top Scores:", global_top_scores)
    print("Global Top Indices:", global_top_indices)
    
    all_results = []
    batch_ids = [101, 102, 103, 104, 105]
    corpus_ids = [f"C{i}" for i in range(num_corpus)]
    
    for i in range(batch_size):
        q_id = batch_ids[i]
        valid_mask = global_top_indices[i] != -1
        final_scores = global_top_scores[i][valid_mask]
        final_indices = global_top_indices[i][valid_mask]
        
        if len(final_scores) > 0:
            sort_idx = np.argsort(-final_scores)
            sorted_scores = final_scores[sort_idx]
            sorted_indices = final_indices[sort_idx]
            
            for rank, (score, c_idx) in enumerate(zip(sorted_scores, sorted_indices), start=1):
                cand_id = corpus_ids[c_idx]
                all_results.append({
                    "s1_id": q_id,
                    "s2_id": cand_id,
                    "rank": rank,
                    "score": score
                })
                
    print(pd.DataFrame(all_results))

mock_batch_retrieve()
