import pandas as pd
import numpy as np
from thefuzz import fuzz
from typing import Dict, Any

def get_token_set(text: str) -> set:
    if not text:
        return set()
    return set(text.split())

def calculate_pairwise_evidence(s1_record: Dict[str, Any], s2_record: Dict[str, Any], idf_dict: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Generates detailed pairwise evidence features between S1 and S2 records.
    """
    features = {}
    
    # 1. Name Features
    name1 = str(s1_record.get('norm_name', ''))
    name2 = str(s2_record.get('norm_name', ''))
    
    if name1 and name2:
        features['name_exact_match'] = int(name1 == name2)
        features['name_fuzz_ratio'] = fuzz.ratio(name1, name2) / 100.0
        features['name_fuzz_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
        features['name_fuzz_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
        
        t1, t2 = get_token_set(name1), get_token_set(name2)
        intersection = t1.intersection(t2)
        features['name_token_jaccard'] = len(intersection) / len(t1.union(t2)) if t1.union(t2) else 0.0
        
        if idf_dict:
            # Information Value (Phase 11)
            idf_intersection = sum(idf_dict.get(t, 0.0) for t in intersection)
            idf_union = sum(idf_dict.get(t, 0.0) for t in t1.union(t2))
            features['name_idf_jaccard'] = idf_intersection / idf_union if idf_union > 0 else 0.0
    else:
        features['name_exact_match'] = 0
        features['name_fuzz_ratio'] = 0.0
        features['name_fuzz_token_set'] = 0.0
        features['name_fuzz_token_sort'] = 0.0
        features['name_token_jaccard'] = 0.0
        features['name_idf_jaccard'] = 0.0

    # 2. Address Features
    addr1 = str(s1_record.get('norm_address', ''))
    addr2 = str(s2_record.get('norm_address', ''))
    
    if addr1 and addr2:
        features['addr_exact_match'] = int(addr1 == addr2)
        features['addr_fuzz_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
        
        t1, t2 = get_token_set(addr1), get_token_set(addr2)
        intersection = t1.intersection(t2)
        features['addr_token_jaccard'] = len(intersection) / len(t1.union(t2)) if t1.union(t2) else 0.0
    else:
        features['addr_exact_match'] = 0
        features['addr_fuzz_token_set'] = 0.0
        features['addr_token_jaccard'] = 0.0

    # 3. Numeric Features (Address numbers)
    num1 = get_token_set(str(s1_record.get('fp_address_numbers', '')))
    num2 = get_token_set(str(s2_record.get('fp_address_numbers', '')))
    
    if num1 and num2:
        features['num_exact_agreement'] = int(num1 == num2)
        features['num_overlap'] = len(num1.intersection(num2)) / len(num1.union(num2))
        features['num_conflict'] = int(len(num1.intersection(num2)) == 0) # Disagreement in numbers
    else:
        features['num_exact_agreement'] = 0
        features['num_overlap'] = 0.0
        features['num_conflict'] = 0

    # 4. Location Features
    c1 = str(s1_record.get('norm_country', ''))
    c2 = str(s2_record.get('norm_country', ''))
    
    if c1 and c2:
        features['country_match'] = int(c1 == c2)
        features['country_conflict'] = int(c1 != c2)
    else:
        features['country_match'] = 0
        features['country_conflict'] = 0

    # 5. Missingness vs Contradiction logic (Phase 10)
    features['name_missing'] = int(not name1 or not name2)
    features['addr_missing'] = int(not addr1 or not addr2)
    features['country_missing'] = int(not c1 or not c2)
    
    # Contradiction means both are present, but heavily dissimilar
    features['name_contradiction'] = int(not features['name_missing'] and features['name_fuzz_token_set'] < 0.3)
    features['addr_contradiction'] = int(not features['addr_missing'] and features['addr_fuzz_token_set'] < 0.3)
    
    return features

def build_idf_dict(corpus_series: pd.Series) -> Dict[str, float]:
    """
    Builds IDF weights for tokens to value rare tokens more.
    """
    from collections import Counter
    import math
    
    N = len(corpus_series)
    doc_freqs = Counter()
    
    for text in corpus_series.dropna():
        tokens = set(str(text).split())
        for t in tokens:
            doc_freqs[t] += 1
            
    idf_dict = {t: math.log(N / (df + 1)) for t, df in doc_freqs.items()}
    return idf_dict

def generate_evidence_dataset(candidates_df: pd.DataFrame, s1_df: pd.DataFrame, s2_s3_df: pd.DataFrame, idf_dict: Dict[str, float] = None) -> pd.DataFrame:
    """
    Takes a dataframe of (s1_id, s2_id) candidates, merges the full records from s1 and s2_s3,
    and calculates all pairwise evidence features.
    """
    print("Merging candidate pairs with source data...")
    # Select only necessary columns to save memory
    needed_cols = ['entity_id', 'norm_name', 'norm_address', 'norm_country', 'fp_address_numbers']
    s1_sub = s1_df[[c for c in needed_cols if c in s1_df.columns]].set_index('entity_id')
    s2_s3_sub = s2_s3_df[[c for c in needed_cols if c in s2_s3_df.columns]].set_index('entity_id')
    
    # We will iterate row by row since we have complex dictionary-based logic
    # To speed this up, we convert the sub dataframes to dicts
    print("Converting data to dictionary lookup...")
    s1_dict = s1_sub.to_dict('index')
    s2_dict = s2_s3_sub.to_dict('index')
    
    records = []
    
    # Try importing tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(candidates_df.itertuples(index=False), total=len(candidates_df), desc="Calculating Evidence")
    except ImportError:
        iterator = candidates_df.itertuples(index=False)
        
    for row in iterator:
        s1_id = row.s1_id
        s2_id = row.s2_id
        
        # Get raw records
        rec1 = s1_dict.get(s1_id, {})
        rec2 = s2_dict.get(s2_id, {})
        
        # Calculate features
        feats = calculate_pairwise_evidence(rec1, rec2, idf_dict)
        
        # Add identifiers and retrieval scores
        feats['s1_id'] = s1_id
        feats['s2_id'] = s2_id
        if hasattr(row, 'rrf_score'):
            feats['rrf_score'] = row.rrf_score
        if hasattr(row, 'score'):
            feats['retrieval_score'] = row.score
        if hasattr(row, 'fused_rank'):
            feats['rank'] = row.fused_rank
            
        records.append(feats)
        
    print("Converting features to DataFrame...")
    return pd.DataFrame(records)
