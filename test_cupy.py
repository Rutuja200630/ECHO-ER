import numpy as np
import scipy.sparse as sp
import time

# Create mock data
vocab_size = 1000
corpus_size = 10000
batch_size = 10

# Dense to sparse
corpus_dense = np.random.rand(corpus_size, vocab_size)
corpus_dense[corpus_dense < 0.95] = 0
corpus_sp = sp.csr_matrix(corpus_dense)

q_dense = np.random.rand(batch_size, vocab_size)
q_dense[q_dense < 0.95] = 0
q_sp = sp.csr_matrix(q_dense)

print("Testing Scipy...")
t0 = time.time()
scores_cpu = q_sp.dot(corpus_sp.T)
print(scores_cpu.shape, scores_cpu.nnz, time.time()-t0)

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    
    print("Testing CuPy...")
    corpus_gpu = cpx_sparse.csr_matrix(corpus_sp)
    corpus_gpu_T = corpus_gpu.T
    
    q_gpu = cpx_sparse.csr_matrix(q_sp)
    
    t0 = time.time()
    scores_gpu = q_gpu.dot(corpus_gpu_T)
    print(scores_gpu.shape, scores_gpu.nnz, time.time()-t0)
    
    scores_back = scores_gpu.get()
    print("Back to CPU:", type(scores_back), scores_back.shape, scores_back.nnz)
except Exception as e:
    print("CuPy not available or error:", e)
