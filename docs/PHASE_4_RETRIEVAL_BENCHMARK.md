# Phase 4: Retrieval Benchmark

## 1. Current Implementation
The current pipeline uses **TF-IDF Sparse Retrieval** (via `scikit-learn TfidfVectorizer`). 
The `main.py` script loads S1 as queries and concatenates S2 and S3 into a single massive corpus dataframe. 
Retrieval iterates row-by-row through the queries using `.iterrows()`. For each query, it performs sparse matrix multiplication against the entire 10-million row index, converts the result to a dense float array using `.toarray().flatten()`, and then runs a full `np.argsort` on all 10 million scores.

## 2. Dataset Sizes
- **S1 (Queries):** 2,206,821 rows
- **S2 + S3 (Corpus):** ~10.1 million rows

## 3. Benchmark Configuration
- **Queries Processed:** 10,000 (S1)
- **Corpus Indexed:** 100,000 (S2 + S3)

## 4. Runtime
- **Index Building (Fit Time):** 1.58 seconds
- **Retrieval Time:** 86.47 seconds (for 10k queries against 100k corpus items)

## 5. Memory Usage
- **Peak RAM (for 100k corpus):** 20.51 MB
- *(Note: A full 10.1M corpus will use ~2-3 GB for the TF-IDF CSR matrix alone, plus DataFrame overhead).*

## 6. Recall@5
0.0070 (0.70%)

## 7. Recall@10
0.0073 (0.73%)

## 8. Recall@20
0.0074 (0.74%)

## 9. Recall@50
0.0074 (0.74%)
*(Note: Recall is artificially extremely low because we truncated the corpus to 100k, meaning 99% of the ground truth matches were not even in the indexed corpus).*

## 10. Correctness Checks
- **Valid IDs:** All retrieved IDs exist in the corpus.
- **Accidental Self-Matches:** None found (S1 queries are correctly matched only against S2/S3 IDs).
- **Duplicate IDs:** Handled properly; `np.argsort` returns unique indices.
- **Missing Names/Empty Strings:** Handled correctly via `.fillna('')` which the Vectorizer gracefully ignores.

## 11. Bottleneck
The fatal bottleneck is calling `.toarray().flatten()` on the sparse output and subsequently using `np.argsort()` to sort the entire dense array of 10 million elements for every single query, inside a slow `.iterrows()` loop.

## 12. Estimated full-dataset runtime
Based on the benchmark, retrieving 1 query against a 100k corpus takes ~8.6 milliseconds. Against a 10M corpus, `np.argsort` takes ~100x longer (~860 ms per query). 
- 860 ms * 2.2 million queries = 1,892,000 seconds = **~21.9 Days**.

## 13. Estimated full-dataset memory
- The TF-IDF sparse matrix for 10M rows will take ~2-3 GB.
- Converting a 10M sparse row to a dense array (`.toarray()`) will consume an additional ~80 MB per query.
- Total memory needed is well within Kaggle's 30 GB limit (estimated ~8 GB total).

## 14. Recommendation for whether the current implementation is Kaggle-feasible
**NO.** The current row-by-row iteration with dense sorting is completely unfeasible for Kaggle's 12-hour timeout. It must be optimized to use batch matrix multiplication and sparse top-K selection (e.g., `scipy.sparse` argpartition or similar sparse-native methods).
