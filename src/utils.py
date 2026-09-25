import pandas as pd
import os

def create_submission_files(matching_results_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame, output_dir: str = '../output'):
    """
    Phase 30: Submission
    Creates the final matching_results.tsv and candidate_pairs.tsv files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    match_path = os.path.join(output_dir, 'matching_results.tsv')
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    matching_results_df.to_csv(match_path, sep='\t', index=False)
    candidate_pairs_df.to_csv(cand_path, sep='\t', index=False)
    
    print(f"Saved matching results to {match_path}")
    print(f"Saved candidate pairs to {cand_path}")

def check_submission(match_path: str, cand_path: str):
    """
    Simple validation matching the challenge constraints.
    """
    match_df = pd.read_csv(match_path, sep='\t', dtype=str).fillna('')
    cand_df = pd.read_csv(cand_path, sep='\t', dtype=str).fillna('')
    
    assert list(match_df.columns) == ['source1_entity_id', 'matched_entity_ids'], "Match columns incorrect."
    assert list(cand_df.columns) == ['source1_entity_id', 'candidate_entity_ids'], "Candidate columns incorrect."
    
    print("Basic submission format checks passed.")
