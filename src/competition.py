import pandas as pd
import numpy as np

def compute_competition_features(features_df: pd.DataFrame, score_col: str = 'rrf_score') -> pd.DataFrame:
    """
    Candidate Competition (Phase 14)
    Compares candidates against one another for the same S1.
    Requires features_df to have 's1_id' and 's2_id' and a relevance score (e.g. rrf_score or LightGBM pred).
    """
    df = features_df.copy()
    
    # Sort by score descending within each S1
    df = df.sort_values(['s1_id', score_col], ascending=[True, False])
    
    # Rank
    df['candidate_rank'] = df.groupby('s1_id')[score_col].rank("dense", ascending=False)
    
    # Get top 1, 2, 3 scores for each S1
    top_scores = df.groupby('s1_id')[score_col].nlargest(3).reset_index()
    
    # Pivot to get top1, top2, top3 columns
    # We will do a simpler approach:
    def get_top_k(group):
        vals = group.values
        return pd.Series({
            'top1_score': vals[0] if len(vals) > 0 else 0,
            'top2_score': vals[1] if len(vals) > 1 else 0,
            'top3_score': vals[2] if len(vals) > 2 else 0,
        })
        
    group_tops = df.groupby('s1_id')[score_col].apply(get_top_k).unstack().reset_index()
    
    df = df.merge(group_tops, on='s1_id', how='left')
    
    # Margins
    df['top1_top2_margin'] = df['top1_score'] - df['top2_score']
    df['top1_top3_margin'] = df['top1_score'] - df['top3_score']
    
    # Candidate density (how many candidates above a threshold)
    # Let's say threshold is mean of top 3
    df['mean_top3'] = (df['top1_score'] + df['top2_score'] + df['top3_score']) / 3.0
    
    # Just basic competition features
    return df

def compute_popularity_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Candidate Popularity / Bridge Risk (Phase 15)
    Measures how often each S2 candidate appears across all S1 candidate pools.
    """
    df = features_df.copy()
    
    # candidate_frequency: How many different S1s retrieved this S2?
    freq = df.groupby('s2_id')['s1_id'].nunique().reset_index()
    freq.columns = ['s2_id', 'candidate_frequency']
    
    df = df.merge(freq, on='s2_id', how='left')
    
    # Inverse candidate frequency
    # We want to penalize candidates that show up for 1000s of different S1s
    total_s1 = df['s1_id'].nunique()
    df['inverse_candidate_frequency'] = np.log((total_s1 + 1) / (df['candidate_frequency'] + 1))
    
    return df
