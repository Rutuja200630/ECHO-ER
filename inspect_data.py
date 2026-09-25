import pandas as pd
import json
import os

base_path = r"c:/Users/Rutuja Hirudkar/OneDrive/c/amazon ml/student_resource/dataset"
train_path = os.path.join(base_path, "train")

files = {
    "train_s1": os.path.join(train_path, "train_source1.tsv"),
    "train_s2": os.path.join(train_path, "train_source2.tsv"),
    "train_s3": os.path.join(train_path, "train_source3.tsv"),
    "train_gt": os.path.join(train_path, "train_ground_truth.tsv")
}

def get_stats(path, out_file):
    out_file.write(f"--- Stats for {path} ---\n")
    df = pd.read_csv(path, sep='\t', dtype=str)
    out_file.write(f"Shape: {df.shape}\n")
    out_file.write(f"Missingness:\n{df.isna().sum()}\n")
    out_file.write(f"Duplicate IDs: {df.duplicated(subset=[df.columns[0]]).sum()}\n\n")
    return df

with open("dataset_stats.txt", "w", encoding="utf-8") as f:
    s1 = get_stats(files["train_s1"], f)
    s2 = get_stats(files["train_s2"], f)
    s3 = get_stats(files["train_s3"], f)
    gt = get_stats(files["train_gt"], f)
    
    # check ground truth multi-matches
    gt['match_count'] = gt['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    f.write(f"Ground truth total matches: {gt['match_count'].sum()}\n")
    f.write(f"Ground truth match distribution:\n{gt['match_count'].value_counts().head(10)}\n")

    # check test set
    test_path = os.path.join(base_path, "test")
    test_files = {
        "test_s1": os.path.join(test_path, "test_source1.tsv"),
        "test_s2": os.path.join(test_path, "test_source2.tsv"),
        "test_s3": os.path.join(test_path, "test_source3.tsv"),
    }
    for name, path in test_files.items():
        if os.path.exists(path):
            get_stats(path, f)


