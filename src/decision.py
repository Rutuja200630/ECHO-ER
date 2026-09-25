import pandas as pd
import numpy as np

def singleton_gate(df: pd.DataFrame, prob_col: str = 'pred', threshold: float = 0.5) -> pd.DataFrame:
    """
    Phase 24: Singleton / No-Match Gate
    Takes DataFrame with scored candidates and returns final matched lists.
    
    The F0.5 metric heavily penalizes false merges, so we only match if probability > threshold.
    Also handles S1s that have NO matches (singletons).
    """
    # Keep only those above threshold
    matches = df[df[prob_col] >= threshold].copy()
    
    # Group by S1 to get comma-separated list of S2s
    # Note: S2/S3 IDs are unique and we shouldn't have duplicates in the list
    match_lists = matches.groupby('s1_id')['s2_id'].apply(lambda x: ','.join(sorted(list(set(x))))).reset_index()
    match_lists.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # We must include EVERY S1 ID in the final output, even if they have no matches
    # This requires merging back with the unique set of S1 IDs
    all_s1 = pd.DataFrame({'source1_entity_id': df['s1_id'].unique()})
    final_output = all_s1.merge(match_lists, on='source1_entity_id', how='left')
    
    # Fill NaN with empty string (Singletons)
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna('')
    
    return final_output

def optimize_threshold(df_val: pd.DataFrame, gt_val: pd.DataFrame, prob_col: str = 'pred'):
    """
    Phase 25: Optimize final decision threshold for F0.5
    """
    best_f05 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.3, 0.95, 0.05):
        pred_out = singleton_gate(df_val, prob_col, threshold=thresh)
        
        # Merge with GT and compute macro F0.5
        # This requires a proper metric computation function matching the Kaggle evaluation
        # We will stub this out for now
        pass
        
    return best_thresh
