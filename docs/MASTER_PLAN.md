# ECHO-ER: Evidence-Consistent Hybrid Optimization for Entity Resolution

## Architecture Philosophy
"Don't just find the most similar record. Find the candidate whose evidence is strongest, whose competitors are weaker, and whose relationship is globally consistent."

## Project Overview
This project builds a high-quality Entity Resolution system that maps records between Source 1 (S1) and Source 2 (S2). 
The final architecture should remain fundamentally:
**HYBRID RETRIEVAL → EVIDENCE ENGINE → LIGHTGBM → CANDIDATE COMPETITION → SINGLETON / NO-MATCH GATE**

## Phased Execution Checklist

### STAGE A: Inspect → Understand → Report
- [x] **Phase 0: Understand the Challenge**
  - Read documentation and project goals.
  - Understand S1, S2, one-to-one or one-to-many match definition, evaluation metrics, and constraints.
- [x] **Phase 1: Data Discovery**
  - Inspect every relevant data file (S1, S2, Train, Test).
  - Calculate row counts, missingness, dtypes, duplicates.
  - Determine ID uniqueness.

### STAGE B: Implement Data Pipeline → Test
- [x] **Phase 2: Data Preprocessing**
  - Unicode/lowercase/whitespace/punctuation normalization.
  - Maintain `original_value` and `normalized_value`.
- [x] **Phase 3: Record Quality**
  - Feature engineering for record completeness (non-null fields, token counts, etc.).
- [x] **Phase 4: Record Fingerprint**
  - Extract structural fingerprints (rare tokens, numbers, postal codes, token counts).

### STAGE C: Implement Retrieval → Test Recall@K
- [x] **Phase 5: Multi-View Representation**
  - Construct NAME VIEW, ADDRESS VIEW, FULL VIEW.
- [x] **Phase 6: Sparse Retrieval**
  - Implement BM25 and TF-IDF baselines.
  - Calculate retrieval recall (Recall@1 to @50).
- [x] **Phase 7: Dense Retrieval**
  - Implement Qwen embeddings (fallback if restricted in Kaggle).
- [x] **Phase 8: Candidate Fusion**
  - Combine sparse and dense retrieval using Reciprocal Rank Fusion (RRF).
- [ ] **Phase 9: Adaptive K**
  - Implement dynamic candidate pool sizing based on confidence/ambiguity.

### STAGE D: Implement Evidence Features → Test
- [x] **Phase 10: Pairwise Evidence Engine**
  - Generate Name, Address, Numeric, Location, Missingness, and Retrieval features.
- [x] **Phase 11: Information Value**
  - IDF-weighted token overlap (rare tokens matter more).
- [x] **Phase 12: Retriever Disagreement**
  - Use sparse/dense disagreement as uncertainty features.
- [x] **Phase 13: Mutual Retrieval**
  - Optional: S1 retrieves S2 AND S2 retrieves S1.

### STAGE E: Implement LightGBM → Evaluate
- [x] **Phase 14: Candidate Competition**
  - Compare candidates for a given S1 (margins, density, rank).
- [x] **Phase 15: Candidate Popularity / Bridge Risk**
  - Measure how often each S2 appears across candidate pools.
- [x] **Phase 16: Training Data Construction**
  - Build pairwise data (positive + hard/random negatives).
- [x] **Phase 17: Train/Validation Split**
  - Ensure leakage-safe validation (group by S1/entity).
- [x] **Phase 18: LightGBM**
  - Train a conservative model with early stopping.
- [x] **Phase 19: Probability Calibration**
  - Optional Platt scaling/isotonic regression.

### STAGE F: Hard-Negative Mining → Retrain
- [x] **Phase 20: Hard Negative Mining**
  - Identify high-probability false positives, add to train set, and retrain.

### STAGE I: Advanced Experiments → Ablation
- [ ] **Phase 21: Optional Counterfactual Evidence**
  - Ablate specific evidence (name/address) to test robustness.
- [ ] **Phase 22: Graph Consistency**
  - Optional: Check node and edge candidate relationships.
- [ ] **Phase 23: Selective Cross-Encoder**
  - Optional: Re-rank only uncertain top candidates.

### STAGE H: Singleton Gate → Optimize F0.5
- [x] **Phase 24: Singleton / No-Match Gate**
  - Final decision layer supporting Match, Ambiguous, No Match.

### STAGE J: Final Model → Submission
- [ ] **Phase 25: Evaluation**
  - Measure F0.5, Precision, Recall, and Recall@K.
- [ ] **Phase 26: Ablation Study**
  - Run controlled experiments baseline to advanced.
- [ ] **Phase 27: Error Analysis**
  - Review top FP/FN and ambiguous cases.
- [ ] **Phase 28: Final Model Selection**
  - Freeze architecture based on validation metrics.
- [x] **Phase 29: Final Training & Prediction**
  - Predict on test set using the exact decision gate.
- [x] **Phase 30: Submission**
  - Generate and validate `submission.csv` according to Kaggle format.

## Final Success Criteria Check
- [x] Dataset automatically loads
- [x] Data schema understood
- [x] Normalization works
- [x] Fingerprints generated
- [x] Multi-view retrieval works
- [x] BM25/TF-IDF retrieval works
- [x] Dense retrieval works or graceful fallback exists
- [x] RRF works
- [ ] Candidate pool generated
- [ ] Retrieval Recall@K calculated
- [x] Evidence features generated
- [x] Missingness handled correctly
- [x] Contradictions handled correctly
- [x] Candidate competition implemented
- [x] LightGBM trained
- [x] Hard negatives mined
- [ ] Validation performed
- [x] Singleton/no-match gate implemented
- [ ] F0.5 evaluated
- [ ] Ablation study completed
- [ ] Error analysis completed
- [ ] Final architecture selected based on evidence
- [ ] Test predictions generated
- [x] submission.csv generated
- [x] Submission format validated
- [x] Documentation generated
