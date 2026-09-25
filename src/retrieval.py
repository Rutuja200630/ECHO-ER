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

def batch_retrieve(queries_df: pd.DataFrame, retriever, q_id_col: str, q_text_col: str, view_name: str, k: int = 20, batch_size: int = 1000) -> pd.DataFrame:
    all_results = []
    
    if isinstance(retriever, OldSparseRetriever):
        for _, row in queries_df.iterrows():
            q_id = row[q_id_col]
            q_text = row[q_text_col]
            candidates = retriever.retrieve(q_text, k=k)
            for c in candidates:
                all_results.append({"s1_id": q_id, "s2_id": c['candidate_id'], "view": view_name, "rank": c['rank'], "score": c['score']})
        return pd.DataFrame(all_results)
    
    # Optimized batch retrieval
    query_texts = queries_df[q_text_col].tolist()
    query_ids = queries_df[q_id_col].tolist()
    
    # Process in batches to limit memory usage
    for start_idx in range(0, len(query_texts), batch_size):
        end_idx = min(start_idx + batch_size, len(query_texts))
        batch_texts = query_texts[start_idx:end_idx]
        batch_ids = query_ids[start_idx:end_idx]
        
        # Sparse matrix of query embeddings (BatchSize x VocabSize)
        q_vecs = retriever.vectorizer.transform(batch_texts)
        
        # Dot product with corpus transpose -> (BatchSize x CorpusSize)
        scores_mat = q_vecs.dot(retriever.matrix.T)
        
        # Iterate over rows in the sparse scores matrix
        for i in range(scores_mat.shape[0]):
            q_id = batch_ids[i]
            row_start = scores_mat.indptr[i]
            row_end = scores_mat.indptr[i+1]
            
            if row_start == row_end:
                continue # No matches
                
            row_data = scores_mat.data[row_start:row_end]
            row_indices = scores_mat.indices[row_start:row_end]
            
            num_non_zero = len(row_data)
            
            if num_non_zero > k:
                # Get indices of top K in row_data
                top_k_local = np.argpartition(row_data, -k)[-k:]
                # Sort these top K precisely
                sorted_local = top_k_local[np.argsort(-row_data[top_k_local])]
            else:
                sorted_local = np.argsort(-row_data)
                
            rank = 1
            for local_idx in sorted_local:
                score = float(row_data[local_idx])
                if score > 0:
                    corpus_idx = row_indices[local_idx]
                    cand_id = retriever.corpus_ids[corpus_idx]
                    all_results.append({"s1_id": q_id, "s2_id": cand_id, "view": view_name, "rank": rank, "score": score})
                    rank += 1
                    
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
