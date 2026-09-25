import time
import numpy as np
import scipy.sparse as sp

# Mock corpus matrix (10,000,000 docs, 100,000 vocab)
num_docs = 1000000
num_vocab = 50000
nnz_per_doc = 5
print("Building mock matrix...")
row_ind = np.repeat(np.arange(num_docs), nnz_per_doc)
col_ind = np.random.randint(0, num_vocab, size=num_docs * nnz_per_doc)
data = np.random.rand(num_docs * nnz_per_doc)

doc_term_mat = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_docs, num_vocab))
inverted_index = doc_term_mat.T.tocsr()

num_queries = 1000
print(f"Testing {num_queries} queries...")

t0 = time.time()
for _ in range(num_queries):
    # simulate a query with 4 tokens
    tokens = np.random.randint(0, num_vocab, size=4)
    
    # Candidate generation
    candidates = set()
    for tok in tokens:
        start = inverted_index.indptr[tok]
        end = inverted_index.indptr[tok+1]
        docs = inverted_index.indices[start:end]
        candidates.update(docs)
        
    candidates = list(candidates)
    if not candidates:
        continue
        
    # Slicing
    cand_mat = doc_term_mat[candidates, :]
    
    # Query vector
    q_vec = sp.csr_matrix((np.ones(4), (np.zeros(4), tokens)), shape=(1, num_vocab))
    
    # Scoring
    scores = q_vec.dot(cand_mat.T).toarray().flatten()
    
    # Top K
    if len(scores) > 50:
        top_idx = np.argpartition(scores, -50)[-50:]
    else:
        top_idx = np.arange(len(scores))

t1 = time.time()
print(f"Time for {num_queries} queries: {t1-t0:.4f} seconds")
