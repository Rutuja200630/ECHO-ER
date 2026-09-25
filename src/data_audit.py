import pandas as pd
import json

def analyze_file(filepath):
    try:
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "missingness": df.isna().sum().to_dict(),
            "duplicates": int(df.duplicated(subset=['entity_id']).sum()) if 'entity_id' in df.columns else int(df.duplicated().sum())
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

def analyze_gt(filepath):
    try:
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "positive_examples": int(df['matched_entity_ids'].notna().sum()),
            "negative_examples": int(df['matched_entity_ids'].isna().sum()),
            "multiple_matches": int((df['matched_entity_ids'].str.split(',').str.len() > 1).sum()),
            "duplicate_pairs_check": int(df.duplicated(subset=['source1_entity_id']).sum()) if 'source1_entity_id' in df.columns else 0
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

files = {
    "S1_Train": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\train\train_source1.tsv",
    "S2_Train": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\train\train_source2.tsv",
    "S3_Train": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\train\train_source3.tsv",
    "GT_Train": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\train\train_ground_truth.tsv",
    "S1_Test": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\test\test_source1.tsv",
    "S2_Test": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\test\test_source2.tsv",
    "S3_Test": r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\student_resource\dataset\test\test_source3.tsv",
}

results = {}
for k, v in files.items():
    if k == "GT_Train":
        results[k] = analyze_gt(v)
    else:
        results[k] = analyze_file(v)

with open(r"c:\Users\Rutuja Hirudkar\OneDrive\c\amazon ml\ECHO-ER\data_audit_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Data audit completed.")
