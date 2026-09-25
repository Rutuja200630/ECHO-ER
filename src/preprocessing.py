import os
import re
import time
import unicodedata
import pandas as pd
import numpy as np

# Regular expressions for cleaning
RE_PUNCT_NAME = re.compile(r'[^\w\s\-]')
RE_PUNCT_ADDR = re.compile(r'[^\w\s\-,]')
RE_WHITESPACE = re.compile(r'\s+')

# Legal suffixes to normalize (e.g., removing or standardizing them)
# We will just remove them from the end for normalization, or standardize.
# The instructions say "Normalize common legal suffixes" - let's remove them from the end of strings
# or standardize them to a generic space. Stripping them is usually best for ER if they are just noise.
SUFFIXES_PATTERN = re.compile(r'\b(inc\.?|llc|ltd\.?|pvt ltd\.?|private limited|gmbh)\b$', re.IGNORECASE)

# Postal code regex (US ZIP, India PIN, general 5-6 digits)
# Example: US (12345 or 12345-6789), India (123456 or 123 456), UK (various)
# A simple 5 to 6 digit extractor will capture most US and India codes
POSTAL_CODE_PATTERN = re.compile(r'\b(\d{5}(?:[-\s]\d{4})?|\d{6}|\d{3}\s\d{3})\b')

def normalize_name(text: str) -> str:
    if pd.isna(text) or text == "" or text is None:
        return ""
    
    text = str(text)
    # Unicode normalize (NFKD)
    text = unicodedata.normalize('NFKD', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove unnecessary punctuation (keep alphanumeric, space, hyphens)
    text = RE_PUNCT_NAME.sub(' ', text)
    
    # Normalize whitespace
    text = RE_WHITESPACE.sub(' ', text).strip()
    
    # Normalize suffixes (remove from end)
    text = SUFFIXES_PATTERN.sub('', text).strip()
    
    return text

def normalize_address(text: str) -> str:
    if pd.isna(text) or text == "" or text is None:
        return ""
    
    text = str(text)
    # Unicode normalize
    text = unicodedata.normalize('NFKD', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove unnecessary punctuation (keep alphanumeric, spaces, hyphens, commas)
    text = RE_PUNCT_ADDR.sub(' ', text)
    
    # Standardize basic abbreviations
    text = re.sub(r'\bstr\b', 'street', text)
    text = re.sub(r'\bst\b', 'street', text)
    text = re.sub(r'\brd\b', 'road', text)
    text = re.sub(r'\bave\b', 'avenue', text)
    text = re.sub(r'\bblvd\b', 'boulevard', text)
    
    # Normalize whitespace
    text = RE_WHITESPACE.sub(' ', text).strip()
    
    return text

def extract_postal_code(text: str) -> str:
    if pd.isna(text) or text == "" or text is None:
        return ""
    text = str(text)
    match = POSTAL_CODE_PATTERN.search(text)
    if match:
        # return digits only for normalization of the code
        return re.sub(r'[^\d]', '', match.group(0))
    return ""

def normalize_country(text: str) -> str:
    if pd.isna(text) or text == "" or text is None:
        return ""
    text = str(text).lower().strip()
    return text

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    df = chunk.copy()
    
    # Missing address flag (1 if missing/empty string, 0 otherwise)
    df['address_missing'] = df['business_address'].apply(lambda x: 1 if pd.isna(x) or str(x).strip() == "" else 0)
    
    # Normalizations
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['address_norm'] = df['business_address'].apply(normalize_address)
    df['country_norm'] = df['country'].apply(normalize_country)
    
    # Postal code extraction
    df['postal_code'] = df['business_address'].apply(extract_postal_code)
    
    return df

def process_file_in_chunks(input_path: str, output_path: str, chunksize: int = 250_000) -> dict:
    start_time = time.time()
    
    total_in = 0
    total_out = 0
    missing_address_in = 0
    address_missing_flag_count = 0
    postal_code_count = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write in chunks
    first_chunk = True
    
    # Read chunk by chunk
    for chunk in pd.read_csv(input_path, sep='\t', chunksize=chunksize, dtype=str):
        total_in += len(chunk)
        missing_address_in += chunk['business_address'].isna().sum() + (chunk['business_address'] == "").sum()
        
        # Process
        processed_chunk = process_chunk(chunk)
        
        total_out += len(processed_chunk)
        address_missing_flag_count += processed_chunk['address_missing'].sum()
        postal_code_count += (processed_chunk['postal_code'] != "").sum()
        
        # Save to output
        mode = 'w' if first_chunk else 'a'
        header = True if first_chunk else False
        processed_chunk.to_csv(output_path, sep='\t', index=False, mode=mode, header=header)
        
        first_chunk = False
        
    duration = time.time() - start_time
    
    return {
        "input_rows": total_in,
        "output_rows": total_out,
        "missing_address_in": missing_address_in,
        "address_missing_flag_count": address_missing_flag_count,
        "postal_code_count": postal_code_count,
        "duration_seconds": duration
    }

def run_phase1():
    base_dir = r"c:/Users/Rutuja Hirudkar/OneDrive/c/amazon ml/student_resource/dataset/train"
    out_dir = r"c:/Users/Rutuja Hirudkar/OneDrive/c/amazon ml/ECHO-ER/data/processed/train"
    report_path = r"c:/Users/Rutuja Hirudkar/OneDrive/c/amazon ml/ECHO-ER/phase1_normalization_report.txt"
    
    files_to_process = [
        ("train_source1.tsv", os.path.join(base_dir, "train_source1.tsv")),
        ("train_source2.tsv", os.path.join(base_dir, "train_source2.tsv")),
        ("train_source3.tsv", os.path.join(base_dir, "train_source3.tsv"))
    ]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Phase 1 Data Normalization Report ===\n\n")
        
        for name, in_path in files_to_process:
            out_path = os.path.join(out_dir, name.replace(".tsv", "_norm.tsv"))
            
            f.write(f"Processing {name}...\n")
            print(f"Processing {name}...")
            
            stats = process_file_in_chunks(in_path, out_path)
            
            f.write(f"Input Rows: {stats['input_rows']}\n")
            f.write(f"Output Rows: {stats['output_rows']}\n")
            f.write(f"Original Missing Address Count: {stats['missing_address_in']}\n")
            f.write(f"address_missing=1 Count: {stats['address_missing_flag_count']}\n")
            f.write(f"Postal Codes Extracted: {stats['postal_code_count']}\n")
            f.write(f"Processing Time: {stats['duration_seconds']:.2f} seconds\n")
            
            # Validation assertions
            assert stats['input_rows'] == stats['output_rows'], "Row counts do not match!"
            assert stats['missing_address_in'] == stats['address_missing_flag_count'], "Missing address flag logic mismatch!"
            
            # Sample check
            f.write("\nSample:\n")
            sample_df = pd.read_csv(out_path, sep='\t', nrows=3, dtype=str)
            for idx, row in sample_df.iterrows():
                f.write(f"  ID: {row['entity_id']}\n")
                f.write(f"  Orig Name: {row['business_name']}\n")
                f.write(f"  Norm Name: {row['name_norm']}\n")
                f.write(f"  Orig Addr: {row['business_address']}\n")
                f.write(f"  Norm Addr: {row['address_norm']}\n")
                f.write(f"  Postal Code: {row['postal_code']}\n")
                f.write(f"  Addr Missing Flag: {row['address_missing']}\n")
                f.write("-" * 40 + "\n")
            f.write("\n========================================\n\n")

if __name__ == "__main__":
    run_phase1()
