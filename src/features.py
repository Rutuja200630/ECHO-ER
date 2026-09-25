import pandas as pd
import numpy as np
from preprocess import extract_numbers

def calculate_record_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates record-quality features to determine the completeness and 
    richness of the information provided for an entity.
    """
    df = df.copy()
    
    # Base fields we care about
    fields = ['business_name', 'business_address', 'country']
    
    # 1. Number of non-null fields
    df['quality_non_null_count'] = df[fields].notna().sum(axis=1)
    
    # 2. String lengths of normalized versions
    df['quality_name_len'] = df['norm_name'].str.len().fillna(0)
    df['quality_address_len'] = df['norm_address'].str.len().fillna(0)
    
    # 3. Token counts
    df['quality_name_tokens'] = df['norm_name'].apply(lambda x: len(x.split()) if x else 0)
    df['quality_address_tokens'] = df['norm_address'].apply(lambda x: len(x.split()) if x else 0)
    
    # 4. Number count (e.g. house numbers, zip codes)
    df['quality_address_numbers'] = df['norm_address'].apply(lambda x: len(extract_numbers(x).split()) if x else 0)
    
    # 5. Overall completeness score (heuristic)
    # Give weights to name length, address length, and presence of fields
    df['quality_score'] = (
        (df['quality_name_tokens'] > 0).astype(int) * 2 +
        (df['quality_address_tokens'] > 0).astype(int) * 2 +
        (df['quality_address_numbers'] > 0).astype(int) * 1 +
        (df['country'].notna() & (df['country'] != "")).astype(int) * 1
    )
    
    return df

def generate_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create structural fingerprints for retrieval and blocking.
    Extract numbers, character patterns, etc.
    """
    df = df.copy()
    
    # Extract just the numbers from the address (often high-signal for identical addresses)
    df['fp_address_numbers'] = df['norm_address'].apply(extract_numbers)
    
    # Create an alphanumeric fingerprint of the name (remove spaces)
    df['fp_name_alpha'] = df['norm_name'].str.replace(' ', '')
    
    # Create a token-sorted fingerprint
    def token_sort(text):
        if not text:
            return ""
        return " ".join(sorted(text.split()))
        
    df['fp_name_sorted'] = df['norm_name'].apply(token_sort)
    df['fp_address_sorted'] = df['norm_address'].apply(token_sort)
    
    return df
