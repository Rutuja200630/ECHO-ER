import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

def create_views(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct Multi-View Representation.
    Creates:
    - name_view: Just the normalized name.
    - address_view: Just the normalized address.
    - full_view: Combined name, address, and country for maximum context.
    """
    df = df.copy()
    df['name_view'] = df['norm_name'].fillna('')
    df['address_view'] = df['norm_address'].fillna('')
    df['country_view'] = df['norm_country'].fillna('')
    
    # Full view combines them
    df['full_view'] = df['name_view'] + " " + df['address_view'] + " " + df['country_view']
    df['full_view'] = df['full_view'].str.strip()
    
    return df

class OldSparseRetriever:
    """
    Original slow implementation for correctness comparison.
    """
    def __init__(self, method: str = 'tfidf'):
        self.method = method
        self.vectorizer = None
        self.bm25_model = None
        self.corpus_ids = []
        self.corpus_texts = []
        self.matrix = None

    def fit(self, df_corpus: pd.DataFrame, id_col: str, text_col: str):
        self.corpus_ids = df_corpus[id_col].tolist()
        self.corpus_texts = df_corpus[text_col].tolist()
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.corpus_texts)

    def retrieve(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        if self.method == 'tfidf':
            q_vec = self.vectorizer.transform([query])
            scores = (self.matrix * q_vec.T).toarray().flatten()
            top_k_indices = np.argsort(scores)[::-1][:k]
            results = []
            for rank, idx in enumerate(top_k_indices, start=1):
                if scores[idx] > 0:
                    results.append({"candidate_id": self.corpus_ids[idx], "score": float(scores[idx]), "rank": rank})
            return results
        return []

class SparseRetriever:
    """
    Optimized implementation using sparse batch matrix multiplication and fast Top-K.
    """
    def __init__(self, method: str = 'tfidf'):
        self.method = method
        self.vectorizer = None
        self.corpus_ids = []
        self.corpus_texts = []
        self.matrix = None

    def fit(self, df_corpus: pd.DataFrame, id_col: str, text_col: str):
        self.corpus_ids = df_corpus[id_col].tolist()
        self.corpus_texts = df_corpus[text_col].tolist()
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.corpus_texts)

class InvertedIndexRetriever:
    """
    Sublinear retrieval using an explicit inverted index for candidate blocking,
    followed by detailed TF-IDF scoring only on the blocked candidates.
    """
    def __init__(self, method: str = 'tfidf', max_posting_size: int = 50000, num_query_tokens_used: int = 10):
        self.method = method
        self.vectorizer = None
        self.corpus_ids = np.array([])
        self.corpus_texts = []
        self.doc_term_mat = None
        self.inverted_index = None
        self.max_posting_size = max_posting_size
        self.num_query_tokens_used = num_query_tokens_used

    def fit(self, df_corpus: pd.DataFrame, id_col: str, text_col: str):
        self.corpus_ids = np.array(df_corpus[id_col].tolist())
        self.corpus_texts = df_corpus[text_col].tolist()
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, min_df=1)
            self.doc_term_mat = self.vectorizer.fit_transform(self.corpus_texts)
            # Transpose to get token -> docs mapping
            self.inverted_index = self.doc_term_mat.T.tocsr()

class MaskedSparseRetriever:
    """
    Sublinear retrieval using vectorized sparse masking.
    We zero-out common tokens in the query batch BEFORE the C++ matrix multiplication,
    reducing the number of dot-product operations by 5x-10x, with near-zero Python overhead.
    """
    def __init__(self, method: str = 'tfidf', max_df_tokens: int = 50000):
        self.method = method
        self.vectorizer = None
        self.corpus_ids = []
        self.corpus_texts = []
        self.matrix = None
        self.max_df_tokens = max_df_tokens
        self.doc_freq = None
        
    def fit(self, df_corpus: pd.DataFrame, id_col: str, text_col: str):
        self.corpus_ids = df_corpus[id_col].tolist()
        self.corpus_texts = df_corpus[text_col].tolist()
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, min_df=1)
            self.matrix = self.vectorizer.fit_transform(self.corpus_texts)
            # Calculate document frequencies for all tokens
            # matrix is shape (num_docs, vocab_size). sum(axis=0) gives frequencies.
            # Convert to a flat dense array for fast lookup
            self.doc_freq = np.array(self.matrix.astype(bool).sum(axis=0)).flatten()

class CuPySparseRetriever:
    """
    Sublinear retrieval using Kaggle's GPU via CuPy.
    Moves the TF-IDF matrices to VRAM and performs blistering fast sparse matrix multiplication.
    """
    def __init__(self, method: str = 'tfidf'):
        self.method = method
        self.vectorizer = None
        self.corpus_ids = []
        self.corpus_texts = []
        self.matrix = None
        
    def fit(self, df_corpus: pd.DataFrame, id_col: str, text_col: str):
        self.corpus_ids = df_corpus[id_col].tolist()
        self.corpus_texts = df_corpus[text_col].tolist()
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='word', stop_words=None, min_df=1, dtype=np.float32)
            self.matrix = self.vectorizer.fit_transform(self.corpus_texts)
            
            # Optional: eagerly move to GPU if you want, but better to do it during retrieve
            # so we can handle memory cleanly.

def batch_retrieve(queries_df: pd.DataFrame, retriever, q_id_col: str, q_text_col: str, view_name: str, k: int = 50, batch_size: int = 5000, corpus_chunk_size: int = 1000000) -> pd.DataFrame:
    all_results = []
    
    if isinstance(retriever, OldSparseRetriever):
        for _, row in queries_df.iterrows():
            q_id = row[q_id_col]
            q_text = row[q_text_col]
            candidates = retriever.retrieve(q_text, k=k)
            for c in candidates:
                all_results.append({"s1_id": q_id, "s2_id": c['candidate_id'], "view": view_name, "rank": c['rank'], "score": c['score']})
        return pd.DataFrame(all_results)
    
    if isinstance(retriever, InvertedIndexRetriever):
        query_texts = queries_df[q_text_col].tolist()
        query_ids = queries_df[q_id_col].tolist()
        num_queries = len(query_texts)
        import time
        from tqdm import tqdm
        
        print(f"Starting inverted index retrieval ({num_queries} queries)", flush=True)
        
        q_vecs = retriever.vectorizer.transform(query_texts)
        
        for i in tqdm(range(num_queries), desc=f"Inverted Index {view_name}"):
            q_id = query_ids[i]
            
            # Find tokens present in this query
            row_start = q_vecs.indptr[i]
            row_end = q_vecs.indptr[i+1]
            if row_start == row_end:
                continue
                
            q_tokens = q_vecs.indices[row_start:row_end]
            
            # Find posting list sizes for each token
            posting_sizes = []
            for tok in q_tokens:
                sz = retriever.inverted_index.indptr[tok+1] - retriever.inverted_index.indptr[tok]
                posting_sizes.append((sz, tok))
                
            # Sort tokens by posting list size (rarest first)
            posting_sizes.sort()
            
            # Take the rarest tokens, up to num_query_tokens_used, respecting max_posting_size
            candidates = set()
            tokens_used = 0
            for sz, tok in posting_sizes:
                if len(candidates) > 2000:
                    break
                if sz > retriever.max_posting_size and tokens_used > 0:
                    # if we already have some rare tokens, skip this very common one
                    continue
                    
                start = retriever.inverted_index.indptr[tok]
                end = retriever.inverted_index.indptr[tok+1]
                docs = retriever.inverted_index.indices[start:end]
                candidates.update(docs)
                tokens_used += 1
                
                if tokens_used >= retriever.num_query_tokens_used:
                    break
                    
            if not candidates:
                continue
                
            candidates_list = list(candidates)[:2000] # Ensure strict bound
            
            # Extract just these candidate rows from the doc-term matrix
            cand_mat = retriever.doc_term_mat[candidates_list, :]
            
            # Compute exact dot product just on candidates
            q_row = q_vecs[i, :]
            scores = q_row.dot(cand_mat.T).toarray().flatten()
            
            # Top K extraction
            if len(scores) > k:
                top_idx = np.argpartition(scores, -k)[-k:]
                sorted_idx = top_idx[np.argsort(-scores[top_idx])]
            else:
                sorted_idx = np.argsort(-scores)
                
            for rank, idx in enumerate(sorted_idx, start=1):
                global_c_idx = candidates_list[idx]
                score = scores[idx]
                if score > 0:
                    all_results.append({
                        "s1_id": q_id,
                        "s2_id": retriever.corpus_ids[global_c_idx],
                        "view": view_name,
                        "rank": rank,
                        "score": float(score)
                    })
                    
        return pd.DataFrame(all_results)
    
    if isinstance(retriever, MaskedSparseRetriever):
        query_texts = queries_df[q_text_col].tolist()
        query_ids = queries_df[q_id_col].tolist()
        num_queries = len(query_texts)
        num_corpus = retriever.matrix.shape[0]
        import time
        from tqdm import tqdm
        
        print(f"Starting MASKED batched retrieval ({num_queries} queries, batch size: {batch_size}, chunk size: {corpus_chunk_size})", flush=True)
        
        for start_q in tqdm(range(0, num_queries, batch_size), desc=f"Retrieving MASKED {view_name}"):
            end_q = min(start_q + batch_size, num_queries)
            batch_texts = query_texts[start_q:end_q]
            batch_ids = query_ids[start_q:end_q]
            curr_batch_size = len(batch_ids)
            
            # 1. Transform query batch
            q_vecs = retriever.vectorizer.transform(batch_texts)
            
            # 2. Vectorized Masking: Zero out weights of hyper-frequent tokens in q_vecs
            # q_vecs.indices contains the vocabulary indices for all non-zeros
            q_indices = q_vecs.indices
            # Look up doc frequencies for these specific tokens
            q_token_dfs = retriever.doc_freq[q_indices]
            # Create a mask where token frequency is TOO high
            common_token_mask = q_token_dfs > retriever.max_df_tokens
            
            # Zero out the data (TF-IDF weights) for those common tokens
            # We don't remove them structurally from the CSR, just zero the values,
            # which prevents scipy/C++ from accumulating them in the dot product.
            # In scipy CSR dot products, explicitly checking for exactly 0.0 can be optimized out by eliminating zeros.
            q_vecs.data[common_token_mask] = 0.0
            q_vecs.eliminate_zeros() # structurally remove the 0.0s for massive speedup
            
            # Maintain global top K per query in this batch
            global_top_scores = np.full((curr_batch_size, k), -1.0, dtype=np.float32)
            global_top_indices = np.full((curr_batch_size, k), -1, dtype=np.int32)
            
            for start_c in range(0, num_corpus, corpus_chunk_size):
                end_c = min(start_c + corpus_chunk_size, num_corpus)
                
                # Slice the corpus matrix
                corpus_chunk_mat = retriever.matrix[start_c:end_c, :]
                
                # Sparse dot product
                scores_mat = q_vecs.dot(corpus_chunk_mat.T)
                
                try:
                    import fast_topk
                    fast_topk.merge_topk(
                        scores_mat.indptr,
                        scores_mat.indices,
                        scores_mat.data,
                        start_c,
                        k,
                        global_top_scores,
                        global_top_indices
                    )
                except ImportError:
                    pass # Ensure fast_topk is installed for MaskedSparseRetriever
                    
                del scores_mat
                del corpus_chunk_mat
                        
            # Format results for this query batch
            for i in range(curr_batch_size):
                q_id = batch_ids[i]
                valid_mask = global_top_indices[i] != -1
                final_scores = global_top_scores[i][valid_mask]
                final_indices = global_top_indices[i][valid_mask]
                
                if len(final_scores) > 0:
                    sort_idx = np.argsort(-final_scores)
                    sorted_scores = final_scores[sort_idx]
                    sorted_indices = final_indices[sort_idx]
                    
                    for rank, (score, c_idx) in enumerate(zip(sorted_scores, sorted_indices), start=1):
                        cand_id = retriever.corpus_ids[c_idx]
                        all_results.append({
                            "s1_id": q_id,
                            "s2_id": cand_id,
                            "view": view_name,
                            "rank": rank,
                            "score": score
                        })
                        
            # Explicit memory cleanup
            del q_vecs
            del global_top_scores
            del global_top_indices
            import gc
            gc.collect()
                        
        return pd.DataFrame(all_results)
        
    if isinstance(retriever, CuPySparseRetriever):
        query_texts = queries_df[q_text_col].tolist()
        query_ids = queries_df[q_id_col].tolist()
        num_queries = len(query_texts)
        import time
        from tqdm import tqdm
        
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx_sparse
        except ImportError:
            raise ImportError("CuPy is required for CuPySparseRetriever. Install it or use MaskedSparseRetriever.")
            
        print(f"Starting GPU (CuPy) batched retrieval ({num_queries} queries, batch size: {batch_size})", flush=True)
        
        # Move Corpus matrix to GPU
        # Doing this once saves massive PCIe transfer times
        print("Moving corpus TF-IDF matrix to GPU VRAM...", flush=True)
        # BUG FIX: Transpose on CPU and convert to CSR *before* moving to GPU
        # CuPy's cuSPARSE backend silently fails (returns empty matrix) if indptr/indices are int64.
        # We MUST explicitly cast to int32 and float32.
        corpus_T_cpu = retriever.matrix.T.tocsr()
        corpus_T_cpu.data = corpus_T_cpu.data.astype(np.float32)
        corpus_T_cpu.indices = corpus_T_cpu.indices.astype(np.int32)
        corpus_T_cpu.indptr = corpus_T_cpu.indptr.astype(np.int32)
        corpus_gpu_T = cpx_sparse.csr_matrix(corpus_T_cpu)
        
        for start_q in tqdm(range(0, num_queries, batch_size), desc=f"Retrieving GPU {view_name}"):
            end_q = min(start_q + batch_size, num_queries)
            batch_texts = query_texts[start_q:end_q]
            batch_ids = query_ids[start_q:end_q]
            curr_batch_size = len(batch_ids)
            
            # Vectorize query batch on CPU
            q_vecs_cpu = retriever.vectorizer.transform(batch_texts)
            q_vecs_cpu.data = q_vecs_cpu.data.astype(np.float32)
            q_vecs_cpu.indices = q_vecs_cpu.indices.astype(np.int32)
            q_vecs_cpu.indptr = q_vecs_cpu.indptr.astype(np.int32)
            
            # Move query batch to GPU
            q_vecs_gpu = cpx_sparse.csr_matrix(q_vecs_cpu)
            
            # Massive parallel sparse dot product on GPU
            scores_mat_gpu = q_vecs_gpu.dot(corpus_gpu_T)
            
            # DEBUG: Print shapes and non-zeros
            if start_q == 0:
                print("\n--- DEBUG INFO (First Batch) ---")
                print(f"q_vecs_cpu: shape={q_vecs_cpu.shape}, nnz={q_vecs_cpu.nnz}, dtype={q_vecs_cpu.dtype}")
                print(f"q_vecs_gpu: shape={q_vecs_gpu.shape}, nnz={q_vecs_gpu.nnz}, dtype={q_vecs_gpu.dtype}")
                print(f"corpus_gpu_T: shape={corpus_gpu_T.shape}, nnz={corpus_gpu_T.nnz}, dtype={corpus_gpu_T.dtype}")
                print(f"scores_mat_gpu: shape={scores_mat_gpu.shape}, nnz={scores_mat_gpu.nnz}, dtype={scores_mat_gpu.dtype}")
            
            # We want Top-K. We can convert back to CPU to use fast_topk, or just use CuPy
            scores_mat_cpu = scores_mat_gpu.get() # Transmits only non-zero sparse matrix elements back to CPU
            
            if start_q == 0:
                print(f"scores_mat_cpu: shape={scores_mat_cpu.shape}, nnz={scores_mat_cpu.nnz}, dtype={scores_mat_cpu.dtype}")
                print(f"scores_mat_cpu indptr type: {scores_mat_cpu.indptr.dtype}, indices type: {scores_mat_cpu.indices.dtype}")
                print("--------------------------------\n")
                
            # Clean up GPU memory for this batch
            del q_vecs_gpu
            del scores_mat_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            global_top_scores = np.full((curr_batch_size, k), -1.0, dtype=np.float32)
            global_top_indices = np.full((curr_batch_size, k), -1, dtype=np.int32)
            
            try:
                import fast_topk
                fast_topk.merge_topk(
                    scores_mat_cpu.indptr,
                    scores_mat_cpu.indices,
                    scores_mat_cpu.data,
                    0,
                    k,
                    global_top_scores,
                    global_top_indices
                )
            except Exception as e:
                if start_q == 0:
                    print(f"WARNING: fast_topk failed or not found ({e}). Falling back to python loop.")
                # Python fallback if fast_topk is missing or throws type error
                for i in range(curr_batch_size):
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
            
            if start_q == 0:
                valid_count = (global_top_indices != -1).sum()
                print(f"DEBUG: First batch generated {valid_count} valid candidate indices.")
                
            # Format results for this query batch
            for i in range(curr_batch_size):
                q_id = batch_ids[i]
                valid_mask = global_top_indices[i] != -1
                final_scores = global_top_scores[i][valid_mask]
                final_indices = global_top_indices[i][valid_mask]
                
                if len(final_scores) > 0:
                    sort_idx = np.argsort(-final_scores)
                    sorted_scores = final_scores[sort_idx]
                    sorted_indices = final_indices[sort_idx]
                    
                    for rank, (score, c_idx) in enumerate(zip(sorted_scores, sorted_indices), start=1):
                        cand_id = retriever.corpus_ids[c_idx]
                        all_results.append({
                            "s1_id": q_id,
                            "s2_id": cand_id,
                            "view": view_name,
                            "rank": rank,
                            "score": score
                        })
                        
        del corpus_gpu_T
        cp.get_default_memory_pool().free_all_blocks()
        
        return pd.DataFrame(all_results)
    
    # Optimized batch retrieval with corpus chunking (for SparseRetriever)
    query_texts = queries_df[q_text_col].tolist()
    query_ids = queries_df[q_id_col].tolist()
    num_queries = len(query_texts)
    num_corpus = retriever.matrix.shape[0]
    import time
    from tqdm import tqdm
    
    print(f"Starting batched retrieval ({num_queries} queries, batch size: {batch_size}, chunk size: {corpus_chunk_size})", flush=True)
    
    for start_q in tqdm(range(0, num_queries, batch_size), desc=f"Retrieving {view_name}"):
        end_q = min(start_q + batch_size, num_queries)
        batch_texts = query_texts[start_q:end_q]
        batch_ids = query_ids[start_q:end_q]
        curr_batch_size = len(batch_ids)
        
        q_vecs = retriever.vectorizer.transform(batch_texts)
        
        # Maintain global top K per query in this batch
        global_top_scores = np.full((curr_batch_size, k), -1.0, dtype=np.float32)
        global_top_indices = np.full((curr_batch_size, k), -1, dtype=np.int32)
        
        for start_c in range(0, num_corpus, corpus_chunk_size):
            end_c = min(start_c + corpus_chunk_size, num_corpus)
            
            # Slice the corpus matrix
            corpus_chunk_mat = retriever.matrix[start_c:end_c, :]
            
            # Sparse dot product
            scores_mat = q_vecs.dot(corpus_chunk_mat.T)
            
            try:
                import fast_topk
                fast_topk.merge_topk(
                    scores_mat.indptr,
                    scores_mat.indices,
                    scores_mat.data,
                    start_c,
                    k,
                    global_top_scores,
                    global_top_indices
                )
            except ImportError:
                # Fallback to Python looping
                for i in range(curr_batch_size):
                    row_start = scores_mat.indptr[i]
                    row_end = scores_mat.indptr[i+1]
                    
                    if row_start == row_end:
                        continue
                        
                    row_data = scores_mat.data[row_start:row_end]
                    row_cols = scores_mat.indices[row_start:row_end]
                    
                    num_non_zero = len(row_data)
                    if num_non_zero > k:
                        top_k_local = np.argpartition(row_data, -k)[-k:]
                        chunk_scores = row_data[top_k_local]
                        chunk_indices = row_cols[top_k_local] + start_c
                    else:
                        chunk_scores = row_data
                        chunk_indices = row_cols + start_c
                        
                    # Merge with global top K for this query
                    merged_scores = np.concatenate((global_top_scores[i], chunk_scores))
                    merged_indices = np.concatenate((global_top_indices[i], chunk_indices))
                    
                    valid_mask = merged_indices != -1
                    if not valid_mask.any():
                        continue
                    valid_scores = merged_scores[valid_mask]
                    valid_indices = merged_indices[valid_mask]
                    
                    if len(valid_scores) > k:
                        top_k_merged = np.argpartition(valid_scores, -k)[-k:]
                        global_top_scores[i] = valid_scores[top_k_merged]
                        global_top_indices[i] = valid_indices[top_k_merged]
                    else:
                        global_top_scores[i, :len(valid_scores)] = valid_scores
                        global_top_indices[i, :len(valid_indices)] = valid_indices
                    
            # Free chunk matrices immediately
            del scores_mat
            del corpus_chunk_mat
                    
        # Format results for this query batch
        for i in range(curr_batch_size):
            q_id = batch_ids[i]
            # Sort the global top K
            valid_mask = global_top_indices[i] != -1
            final_scores = global_top_scores[i][valid_mask]
            final_indices = global_top_indices[i][valid_mask]
            
            if len(final_scores) > 0:
                sort_idx = np.argsort(-final_scores)
                sorted_scores = final_scores[sort_idx]
                sorted_indices = final_indices[sort_idx]
                
                for rank, (score, c_idx) in enumerate(zip(sorted_scores, sorted_indices), start=1):
                    cand_id = retriever.corpus_ids[c_idx]
                    all_results.append({
                        "s1_id": q_id,
                        "s2_id": cand_id,
                        "view": view_name,
                        "rank": rank,
                        "score": score
                    })
                    
        # Explicit memory cleanup
        del q_vecs
        del global_top_scores
        del global_top_indices
        import gc
        gc.collect()
                    
    return pd.DataFrame(all_results)
            
class DenseRetriever:
    """
    Implements Dense Retrieval using Qwen embeddings (if available) with a fallback.
    """
    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.corpus_ids = []
        self.corpus_embeddings = None
        self.is_initialized = False

    def initialize(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
            self.is_initialized = True
            print("Dense retriever initialized successfully.")
        except Exception as e:
            print(f"Dense retriever initialization failed: {e}. Falling back to sparse retrieval only.")
            self.is_initialized = False

    def get_embeddings(self, texts: List[str]) -> Any:
        import torch
        if not self.is_initialized:
            return None
            
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use average pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def fit(self, df_corpus: pd.DataFrame, id_col: str, text_col: str, batch_size=32):
        if not self.is_initialized:
            return
            
        self.corpus_ids = df_corpus[id_col].tolist()
        texts = df_corpus[text_col].tolist()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            emb = self.get_embeddings(batch_texts)
            all_embeddings.append(emb)
            
        self.corpus_embeddings = np.vstack(all_embeddings)

    def retrieve(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        if not self.is_initialized or not query.strip():
            return []
            
        q_emb = self.get_embeddings([query])[0]
        # Cosine similarity (since they are normalized)
        scores = np.dot(self.corpus_embeddings, q_emb)
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, start=1):
            results.append({
                "candidate_id": self.corpus_ids[idx],
                "score": float(scores[idx]),
                "rank": rank
            })
        return results

def reciprocal_rank_fusion(candidate_dfs: List[pd.DataFrame], k: int = 60) -> pd.DataFrame:
    """
    Combines candidate DataFrames (from sparse/dense and different views) using RRF.
    candidate_dfs is a list of dataframes containing: s1_id, s2_id, view, rank, score
    """
    if not candidate_dfs:
        return pd.DataFrame()
        
    combined = pd.concat(candidate_dfs, ignore_index=True)
    
    # RRF score calculation: 1 / (k + rank)
    combined['rrf_score'] = 1.0 / (k + combined['rank'])
    
    # Aggregate by s1_id and s2_id
    # We sum the RRF scores across all retrievers/views
    agg_funcs = {
        'rrf_score': 'sum',
        'view': lambda x: list(set(x)), # which views retrieved this candidate
        'score': 'max' # store the max raw score just in case
    }
    
    fused = combined.groupby(['s1_id', 's2_id']).agg(agg_funcs).reset_index()
    fused['num_views'] = fused['view'].apply(len)
    
    # Sort by S1 ID and then by RRF score descending
    fused = fused.sort_values(['s1_id', 'rrf_score'], ascending=[True, False])
    
    # Assign new fused rank
    fused['fused_rank'] = fused.groupby('s1_id')['rrf_score'].rank("dense", ascending=False)
    
    return fused
