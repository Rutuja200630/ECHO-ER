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
    
    # Optimized batch retrieval with corpus chunking
    query_texts = queries_df[q_text_col].tolist()
    query_ids = queries_df[q_id_col].tolist()
    num_queries = len(query_texts)
    num_corpus = retriever.matrix.shape[0]
    import time
    
    for start_q in range(0, num_queries, batch_size):
        if start_q % (batch_size * 5) == 0:
            print(f"  -> Processing query {start_q}/{num_queries} ({(start_q/num_queries)*100:.1f}%)", flush=True)
            
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
            
            # Extract top K for each query in this chunk
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
                    chunk_indices = row_cols[top_k_local] + start_c # map to global corpus index
                else:
                    chunk_scores = row_data
                    chunk_indices = row_cols + start_c
                    
                # Merge with global top K for this query
                merged_scores = np.concatenate((global_top_scores[i], chunk_scores))
                merged_indices = np.concatenate((global_top_indices[i], chunk_indices))
                
                # Get the new top K from the merged array
                # Filter out -1 defaults
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
