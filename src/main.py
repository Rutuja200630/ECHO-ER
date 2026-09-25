import os
import argparse
import pandas as pd
from preprocess import normalize_dataframe
from features import calculate_record_quality, generate_fingerprints
from retrieval import create_views, SparseRetriever, MaskedSparseRetriever, DenseRetriever, batch_retrieve, reciprocal_rank_fusion
from evidence import calculate_pairwise_evidence, build_idf_dict
from competition import compute_competition_features, compute_popularity_features
from train import construct_training_data, train_lightgbm, get_hard_negatives
from decision import singleton_gate
from utils import create_submission_files, check_submission
import data_audit

def run_phase(phase: str, data_dir: str, output_dir: str, start_row: int = None, end_row: int = None):
    print(f"Starting ECHO-ER Pipeline - Phase: {phase}")
    
    if phase == "audit":
        print("Running Data Audit...")
        # Since data audit imports aren't fully integrated here, we can call it directly
        # For simplicity, assuming data_audit handles its own paths or we adjust it.
        # But for now, we just print
        print("Audit is usually run locally. See data_audit.py")
        
    elif phase == "preprocess":
        print("Running Preprocessing...")
        for source in ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']:
            path = os.path.join(data_dir, source)
            if not os.path.exists(path):
                print(f"Skipping {source}, file not found at {path}")
                continue
            
            print(f"Loading {source}...")
            df = pd.read_csv(path, sep='\t', dtype=str)
            
            print(f"Normalizing {source}...")
            df_norm = normalize_dataframe(df)
            
            out_path = os.path.join(output_dir, f"norm_{source}")
            print(f"Saving to {out_path}...")
            df_norm.to_csv(out_path, sep='\t', index=False)
            
        print("Preprocessing complete.")
        
    elif phase == "features":
        print("Running Feature Engineering (Phases 3 & 4)...")
        for source in ['norm_train_source1.tsv', 'norm_train_source2.tsv', 'norm_train_source3.tsv']:
            path = os.path.join(output_dir, source)
            if not os.path.exists(path):
                print(f"Skipping {source}, file not found at {path}. Did you run --phase preprocess?")
                continue
                
            print(f"Loading {source}...")
            df = pd.read_csv(path, sep='\t', dtype=str)
            
            # Convert NaN back to empty strings where needed, otherwise len() might break if NaNs sneaked in
            df['norm_name'] = df['norm_name'].fillna("")
            df['norm_address'] = df['norm_address'].fillna("")
            
            print(f"Calculating record quality for {source}...")
            df_feat = calculate_record_quality(df)
            
            print(f"Generating fingerprints for {source}...")
            df_feat = generate_fingerprints(df_feat)
            
            out_name = source.replace('norm_', 'feat_')
            out_path = os.path.join(output_dir, out_name)
            print(f"Saving to {out_path}...")
            df_feat.to_csv(out_path, sep='\t', index=False)
            
        print("Feature engineering complete.")
        
    elif phase == "retrieval":
        print("Running Retrieval (Phases 5-8)...")
        # Load queries (S1)
        s1_path = os.path.join(output_dir, 'feat_train_source1.tsv')
        if not os.path.exists(s1_path):
            print(f"Skipping retrieval, query file not found at {s1_path}. Did you run --phase features?")
            return
            
        print("Loading queries (S1) and corpus (S2/S3)...")
        df_s1 = pd.read_csv(s1_path, sep='\t', dtype=str)
        
        if start_row is not None or end_row is not None:
            _start = start_row if start_row is not None else 0
            _end = end_row if end_row is not None else len(df_s1)
            print(f"Slicing S1 queries from {_start} to {_end}")
            df_s1 = df_s1.iloc[_start:_end]
            
        # Load and combine corpus (S2 + S3)
        df_s2 = pd.read_csv(os.path.join(output_dir, 'feat_train_source2.tsv'), sep='\t', dtype=str)
        df_s3 = pd.read_csv(os.path.join(output_dir, 'feat_train_source3.tsv'), sep='\t', dtype=str)
        df_corpus = pd.concat([df_s2, df_s3], ignore_index=True)
        
        print(f"Total S1 Queries: {len(df_s1)}")
        print(f"Total S2/S3 Corpus: {len(df_corpus)}")
        
        print("Creating Multi-View Representations...")
        df_s1 = create_views(df_s1)
        df_corpus = create_views(df_corpus)
        
        candidate_dfs = []
        
        print("Building TF-IDF Retriever for Name View...", flush=True)
        retriever_name = MaskedSparseRetriever(method='tfidf', max_df_tokens=50000)
        retriever_name.fit(df_corpus, id_col='entity_id', text_col='name_view')
        
        print("Retrieving candidates based on Name View...")
        res_name = batch_retrieve(df_s1, retriever_name, 'entity_id', 'name_view', 'name_view', k=50, batch_size=5000, corpus_chunk_size=1000000)
        if not res_name.empty:
            candidate_dfs.append(res_name)
            
        print("Building TF-IDF Retriever for Address View...", flush=True)
        retriever_addr = MaskedSparseRetriever(method='tfidf', max_df_tokens=50000)
        retriever_addr.fit(df_corpus, id_col='entity_id', text_col='address_view')
        
        print("Retrieving candidates based on Address View...")
        res_addr = batch_retrieve(df_s1, retriever_addr, 'entity_id', 'address_view', 'address_view', k=50, batch_size=5000, corpus_chunk_size=1000000)
        if not res_addr.empty:
            candidate_dfs.append(res_addr)
            
        print("Fusing candidates with Reciprocal Rank Fusion (RRF)...")
        if candidate_dfs:
            fused_candidates = reciprocal_rank_fusion(candidate_dfs)
            
            suffix = ""
            if start_row is not None or end_row is not None:
                _start_str = start_row if start_row is not None else 0
                _end_str = end_row if end_row is not None else "end"
                suffix = f"_{_start_str}_{_end_str}"
                
            out_path = os.path.join(output_dir, f"fused_candidates{suffix}.tsv")
            print(f"Saving {len(fused_candidates)} fused candidates to {out_path}...")
            fused_candidates.to_csv(out_path, sep='\t', index=False)
        else:
            print("No candidates retrieved.")
            
        print("Retrieval complete.")
        
    elif phase == "evidence":
        print("Running Evidence Engine...")
        print("Evidence generation complete.")
        
    elif phase == "train":
        print("Running Training...")
        print("Training complete.")
        
    elif phase == "predict":
        print("Running Prediction & Submission...")
        print("Submission generated.")
        
    elif phase == "all":
        print("Running full pipeline...")
        
    else:
        print(f"Unknown phase: {phase}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECHO-ER Kaggle Pipeline")
    parser.add_argument("--phase", type=str, default="all", help="Phase to run: audit, preprocess, features, retrieval, evidence, train, predict, all")
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="../output", help="Path to output")
    parser.add_argument("--start_row", type=int, default=None, help="Start row for query slicing")
    parser.add_argument("--end_row", type=int, default=None, help="End row for query slicing")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_phase(args.phase, args.data_dir, args.output_dir, args.start_row, args.end_row)

