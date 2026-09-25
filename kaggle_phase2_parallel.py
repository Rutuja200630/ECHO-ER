import os
import re
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import glob

# ---------------------------------------------------------
# REGEX FOR FINGERPRINTING
# ---------------------------------------------------------
# Find blocks of numbers
RE_NUMBERS = re.compile(r'\d+')
# Find phone-like patterns (10-12 digits, optional plus, optional dashes/spaces)
RE_PHONE = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

def extract_numbers(text: str) -> str:
    """Extract all numeric blocks as a space-separated string"""
    if pd.isna(text) or text == "": return ""
    return " ".join(RE_NUMBERS.findall(str(text)))

def extract_phone(text: str) -> str:
    """Extract first phone-like sequence if present"""
    if pd.isna(text) or text == "": return ""
    match = RE_PHONE.search(str(text))
    return re.sub(r'[^\d]', '', match.group(0)) if match else ""

def compute_quality_score(row: pd.Series) -> float:
    """
    Heuristic record quality:
    - Base 1.0
    - Missing address: -0.4
    - Name very short (< 5 chars): -0.3
    - Address very short (< 10 chars, if present): -0.2
    """
    score = 1.0
    if row.get('address_missing') == 1 or row.get('address_missing') == '1':
        score -= 0.4
    elif len(str(row.get('address_norm', ''))) < 10:
        score -= 0.2
        
    if len(str(row.get('name_norm', ''))) < 5:
        score -= 0.3
        
    return max(0.1, round(score, 2))

def process_fingerprint_chunk(chunk_data):
    """
    Worker function to compute fingerprint features for a chunk.
    """
    chunk_id, chunk = chunk_data
    
    # Text lengths
    chunk['name_char_count'] = chunk['name_norm'].fillna("").astype(str).apply(len)
    chunk['addr_char_count'] = chunk['address_norm'].fillna("").astype(str).apply(len)
    
    # Token counts
    chunk['name_token_count'] = chunk['name_norm'].fillna("").astype(str).apply(lambda x: len(x.split()))
    chunk['addr_token_count'] = chunk['address_norm'].fillna("").astype(str).apply(lambda x: len(x.split()))
    
    # Extract numbers (store as string blocks to compare later)
    chunk['name_numbers'] = chunk['name_norm'].apply(extract_numbers)
    chunk['addr_numbers'] = chunk['address_norm'].apply(extract_numbers)
    
    # Extract phone numbers (often hidden in name or address)
    chunk['name_phone'] = chunk['business_name'].apply(extract_phone)
    chunk['addr_phone'] = chunk['business_address'].apply(extract_phone)
    
    # Compute heuristic quality score
    chunk['record_quality'] = chunk.apply(compute_quality_score, axis=1)
    
    # We only need to save the entity_id and the new fingerprint features to avoid duplicating the huge text columns.
    # Later, we can join them on entity_id.
    features = [
        'entity_id', 'name_char_count', 'addr_char_count', 'name_token_count', 'addr_token_count',
        'name_numbers', 'addr_numbers', 'name_phone', 'addr_phone', 'record_quality'
    ]
    
    return chunk_id, chunk[features]

def run_kaggle_phase2():
    # Detect the processed Phase 1 directory
    processed_dir = "/kaggle/working/data/processed/train"
    out_dir = "/kaggle/working/data/fingerprints/train"
    
    if not os.path.exists(processed_dir):
        print(f"❌ Error: Could not find Phase 1 output at {processed_dir}. Run Phase 1 first!")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    files_to_process = [
        "train_source1_norm.tsv",
        "train_source2_norm.tsv",
        "train_source3_norm.tsv"
    ]
    
    num_cores = multiprocessing.cpu_count()
    print(f"🚀 Initializing Phase 2 Fingerprinting using {num_cores} CPU cores...")
    
    chunksize = 200_000 
    
    for filename in files_to_process:
        in_path = os.path.join(processed_dir, filename)
        if not os.path.exists(in_path):
            print(f"⚠️ Skipping {in_path}, file not found.")
            continue
            
        out_path = os.path.join(out_dir, filename.replace("_norm.tsv", "_fingerprint.tsv"))
        print(f"\nExtracting fingerprints from {filename}...")
        start_time = time.time()
        
        # Read the normalized TSV in chunks
        chunk_iter = pd.read_csv(in_path, sep='\t', chunksize=chunksize, dtype=str)
        chunk_data = [(i, chunk) for i, chunk in enumerate(chunk_iter)]
        
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for result in executor.map(process_fingerprint_chunk, chunk_data):
                results.append(result)
                
        # Sort chunks by chunk_id
        results.sort(key=lambda x: x[0])
        
        # Write extracted fingerprints back to disk
        first = True
        total_rows = 0
        for _, processed_chunk in results:
            total_rows += len(processed_chunk)
            mode = 'w' if first else 'a'
            header = True if first else False
            processed_chunk.to_csv(out_path, sep='\t', index=False, mode=mode, header=header)
            first = False
            
        print(f"✅ Finished {filename} | Extracted {total_rows:,} fingerprints in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_kaggle_phase2()
