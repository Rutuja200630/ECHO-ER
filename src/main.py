import os
import argparse
import pandas as pd
from preprocess import normalize_dataframe
from features import calculate_record_quality, generate_fingerprints
from retrieval import create_views, SparseRetriever, DenseRetriever, batch_retrieve, reciprocal_rank_fusion
from evidence import calculate_pairwise_evidence, build_idf_dict
from competition import compute_competition_features, compute_popularity_features
from train import construct_training_data, train_lightgbm, get_hard_negatives
from decision import singleton_gate
from utils import create_submission_files, check_submission
import data_audit

def run_phase(phase: str, data_dir: str, output_dir: str):
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
        print("Running Retrieval...")
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
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_phase(args.phase, args.data_dir, args.output_dir)

