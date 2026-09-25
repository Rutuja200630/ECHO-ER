import re
import unicodedata
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    Perform aggressive text normalization.
    - lowercase
    - unicode normalization
    - replace specific punctuation with space
    - remove multiple whitespaces
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Unicode normalize
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    
    # Lowercase
    text = text.lower()
    
    # Replace common punctuation with space (keep alphanumeric)
    # Be careful not to destroy numbers or useful tokens
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization to dataframe while keeping original columns.
    """
    df = df.copy()
    
    # We maintain original_value implicitly by adding normalized_value
    if 'business_name' in df.columns:
        df['norm_name'] = df['business_name'].apply(clean_text)
        
    if 'business_address' in df.columns:
        df['norm_address'] = df['business_address'].apply(clean_text)
        
    if 'country' in df.columns:
        df['norm_country'] = df['country'].astype(str).str.lower().str.strip()
        
    return df

def extract_numbers(text: str) -> str:
    """
    Extract all numbers from text (useful for address fingerprinting).
    """
    if not text:
        return ""
    numbers = re.findall(r'\d+', text)
    return " ".join(numbers)

def extract_email(text: str) -> str:
    """
    Extract email if present in the text.
    """
    if not text:
        return ""
    # simple regex for email
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else ""

def extract_phone(text: str) -> str:
    """
    Extract phone-like sequence (very rudimentary for now).
    """
    if not text:
        return ""
    # Look for sequences of digits that could be a phone number
    match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return match.group(0) if match else ""

if __name__ == "__main__":
    # Test normalization
    sample = pd.DataFrame({
        "business_name": ["Apple Inc.", "Samsung (Korea)", "McDonald's", None],
        "business_address": ["1 Infinite Loop, Cupertino, CA 95014", "Seoul, South Korea", "123 Burger St.", ""]
    })
    
    out = normalize_dataframe(sample)
    print(out[['business_name', 'norm_name', 'business_address', 'norm_address']])
