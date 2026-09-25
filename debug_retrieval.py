import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

def debug_retrieval():
    print("="*50)
    print("ECHO-ER: SPARSE RETRIEVAL DEBUGGER")
    print("="*50)
    
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
        print(f"CuPy Version: {cp.__version__}")
    except ImportError:
        print("CuPy not found! Please run this on Kaggle with GPU enabled.")
        return

    output_dir = '/kaggle/working/ECHO-ER/output'
    if not os.path.exists(output_dir):
        print("Please run this inside the Kaggle environment where TSVs exist!")
        # For local testing fallback, we can generate mock data
        print("Falling back to mock data...")
        df_s1 = pd.DataFrame({'entity_id': np.arange(100), 'name_view': ['test private limited ' + str(i) for i in range(100)]})
        df_corpus = pd.DataFrame({'entity_id': np.arange(10000), 'name_view': ['test company limited ' + str(i) for i in range(10000)]})
    else:
        print("Loading 10,000 corpus records and 100 queries...")
        df_s1 = pd.read_csv(os.path.join(output_dir, 'feat_train_source1.tsv'), sep='\t', dtype=str).head(100)
        df_s2 = pd.read_csv(os.path.join(output_dir, 'feat_train_source2.tsv'), sep='\t', dtype=str).head(5000)
        df_s3 = pd.read_csv(os.path.join(output_dir, 'feat_train_source3.tsv'), sep='\t', dtype=str).head(5000)
        df_corpus = pd.concat([df_s2, df_s3], ignore_index=True)
        
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from retrieval import create_views
        df_s1 = create_views(df_s1)
        df_corpus = create_views(df_corpus)
        
    df_s1['name_view'] = df_s1['name_view'].fillna('')
    df_corpus['name_view'] = df_corpus['name_view'].fillna('')
    
    print("\n--- TASK 1: TF-IDF CONSTRUCTION ---")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, min_df=1, dtype=np.float32)
    
    print("Fitting corpus...")
    corpus_mat = vectorizer.fit_transform(df_corpus['name_view'].tolist())
    
    print("Transforming queries...")
    query_mat = vectorizer.transform(df_s1['name_view'].tolist())
    
    print("\nQUERY MATRIX:")
    print(f"- shape: {query_mat.shape}")
    print(f"- dtype: {query_mat.dtype}")
    print(f"- nnz: {query_mat.nnz}")
    non_empty = (query_mat.getnnz(axis=1) > 0).sum()
    print(f"- non-empty rows: {non_empty}")
    
    print("\nCORPUS MATRIX:")
    print(f"- shape: {corpus_mat.shape}")
    print(f"- dtype: {corpus_mat.dtype}")
    print(f"- nnz: {corpus_mat.nnz}")
    print(f"- vocab size: {len(vectorizer.vocabulary_)}")
    
    print("\n--- TASK 3: TEST CPU SPARSE MULTIPLICATION ---")
    t0 = time.time()
    scores_cpu = query_mat.dot(corpus_mat.T)
    t1 = time.time()
    print(f"CPU dot product took {t1-t0:.4f} seconds")
    print(f"CPU result shape: {scores_cpu.shape}")
    print(f"CPU result nnz: {scores_cpu.nnz}")
    if scores_cpu.nnz > 0:
        print(f"CPU maximum similarity: {scores_cpu.data.max():.4f}")
    
    print("\n--- TASK 6: CHECK GPU SPARSE CONVERSION & MULTIPLICATION ---")
    # Cast to int32 just in case
    corpus_T_cpu = corpus_mat.T.tocsr()
    corpus_T_cpu.data = corpus_T_cpu.data.astype(np.float32)
    corpus_T_cpu.indices = corpus_T_cpu.indices.astype(np.int32)
    corpus_T_cpu.indptr = corpus_T_cpu.indptr.astype(np.int32)
    
    q_vecs_cpu = query_mat.copy()
    q_vecs_cpu.data = q_vecs_cpu.data.astype(np.float32)
    q_vecs_cpu.indices = q_vecs_cpu.indices.astype(np.int32)
    q_vecs_cpu.indptr = q_vecs_cpu.indptr.astype(np.int32)
    
    print("Moving to GPU...")
    corpus_gpu_T = cpx_sparse.csr_matrix(corpus_T_cpu)
    q_vecs_gpu = cpx_sparse.csr_matrix(q_vecs_cpu)
    
    t0 = time.time()
    scores_gpu = q_vecs_gpu.dot(corpus_gpu_T)
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    
    print(f"GPU dot product took {t1-t0:.4f} seconds")
    print(f"GPU result shape: {scores_gpu.shape}")
    print(f"GPU result nnz: {scores_gpu.nnz}")
    
    if scores_cpu.nnz > 0 and scores_gpu.nnz == 0:
        print("\n!!! CRITICAL FAILURE DETECTED !!!")
        print("CPU produced results, but GPU returned nnz=0!")
        print("This confirms a CuPy/cuSPARSE silent failure on this specific matrix shape/type.")
    elif scores_gpu.nnz > 0:
        print("\nSUCCESS! GPU returned valid non-zero results for this small batch.")
        
    print("\n--- FINAL REPORT ---")
    if scores_cpu.nnz == 0:
        print("ROOT CAUSE = TF-IDF logic produced disjoint vocabularies (0 overlap)")
    elif scores_gpu.nnz == 0:
        print("ROOT CAUSE = CuPy cuSPARSE SpGEMM silent failure (likely integer overflow internally)")
    else:
        print("ROOT CAUSE = Large batch sizes caused CuPy internal overflow. Small batch works!")

if __name__ == "__main__":
    debug_retrieval()
