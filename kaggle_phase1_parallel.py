import os
import re
import time
import unicodedata
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ---------------------------------------------------------
# REGEX & PREPROCESSING COMPILED PATTERNS
# ---------------------------------------------------------
RE_PUNCT_NAME = re.compile(r'[^\w\s\-]')
RE_PUNCT_ADDR = re.compile(r'[^\w\s\-,]')
RE_WHITESPACE = re.compile(r'\s+')
SUFFIXES_PATTERN = re.compile(r'\b(inc\.?|llc|ltd\.?|pvt ltd\.?|private limited|gmbh)\b$', re.IGNORECASE)
POSTAL_CODE_PATTERN = re.compile(r'\b(\d{5}(?:[-\s]\d{4})?|\d{6}|\d{3}\s\d{3})\b')

# ---------------------------------------------------------
# NORMALIZATION FUNCTIONS
# ---------------------------------------------------------
def normalize_name(text: str) -> str:
    if pd.isna(text) or text == "" or text is None: return ""
    text = unicodedata.normalize('NFKD', str(text)).lower()
    text = RE_PUNCT_NAME.sub(' ', text)
    text = RE_WHITESPACE.sub(' ', text).strip()
    return SUFFIXES_PATTERN.sub('', text).strip()

def normalize_address(text: str) -> str:
    if pd.isna(text) or text == "" or text is None: return ""
    text = unicodedata.normalize('NFKD', str(text)).lower()
    text = RE_PUNCT_ADDR.sub(' ', text)
    text = re.sub(r'\bstr\b', 'street', text)
    text = re.sub(r'\bst\b', 'street', text)
    text = re.sub(r'\brd\b', 'road', text)
    text = re.sub(r'\bave\b', 'avenue', text)
    text = re.sub(r'\bblvd\b', 'boulevard', text)
    return RE_WHITESPACE.sub(' ', text).strip()

def extract_postal_code(text: str) -> str:
    if pd.isna(text) or text == "" or text is None: return ""
    match = POSTAL_CODE_PATTERN.search(str(text))
    return re.sub(r'[^\d]', '', match.group(0)) if match else ""

def normalize_country(text: str) -> str:
    if pd.isna(text) or text == "" or text is None: return ""
    return str(text).lower().strip()

# ---------------------------------------------------------
# PARALLEL CHUNK PROCESSING WORKER
# ---------------------------------------------------------
def process_chunk(chunk_data):
    """
    Worker function to process a single DataFrame chunk in a separate CPU process.
    """
    chunk_id, chunk = chunk_data
    
    # Missing address flag
    chunk['address_missing'] = chunk['business_address'].apply(
        lambda x: 1 if pd.isna(x) or str(x).strip() == "" else 0
    )
    
    # Normalizations
    chunk['name_norm'] = chunk['business_name'].apply(normalize_name)
    chunk['address_norm'] = chunk['business_address'].apply(normalize_address)
    chunk['country_norm'] = chunk['country'].apply(normalize_country)
    chunk['postal_code'] = chunk['business_address'].apply(extract_postal_code)
    
    return chunk_id, chunk

# ---------------------------------------------------------
# MAIN KAGGLE EXECUTION (MAX CPU UTILIZATION)
# ---------------------------------------------------------
def run_kaggle_phase1():
    # Adjust paths for Kaggle environment (usually /kaggle/input/... and /kaggle/working/...)
    # Replace these paths when running in the Kaggle notebook
    base_dir = "/kaggle/input/your-dataset-name/train" 
    out_dir = "/kaggle/working/data/processed/train"
    
    os.makedirs(out_dir, exist_ok=True)
    
    files_to_process = [
        ("train_source1.tsv", os.path.join(base_dir, "train_source1.tsv")),
        ("train_source2.tsv", os.path.join(base_dir, "train_source2.tsv")),
        ("train_source3.tsv", os.path.join(base_dir, "train_source3.tsv"))
    ]
    
    # Maximize CPU usage (Kaggle has 4 cores)
    num_cores = multiprocessing.cpu_count()
    print(f"🚀 Initializing Kaggle Parallel Processing using {num_cores} CPU cores...")
    
    chunksize = 200_000 # Optimal size for RAM vs Speed
    
    for name, in_path in files_to_process:
        if not os.path.exists(in_path):
            print(f"⚠️ Skipping {in_path}, file not found.")
            continue
            
        out_path = os.path.join(out_dir, name.replace(".tsv", "_norm.tsv"))
        print(f"\nProcessing {name} in parallel...")
        start_time = time.time()
        
        # 1. Read the TSV file in chunks
        chunk_iter = pd.read_csv(in_path, sep='\t', chunksize=chunksize, dtype=str)
        chunk_data = [(i, chunk) for i, chunk in enumerate(chunk_iter)]
        
        # 2. Process chunks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Map the worker function to the chunks
            for result in executor.map(process_chunk, chunk_data):
                results.append(result)
                
        # 3. Sort chunks by chunk_id to maintain original order
        results.sort(key=lambda x: x[0])
        
        # 4. Write back to disk
        first = True
        total_rows = 0
        for _, processed_chunk in results:
            total_rows += len(processed_chunk)
            mode = 'w' if first else 'a'
            header = True if first else False
            processed_chunk.to_csv(out_path, sep='\t', index=False, mode=mode, header=header)
            first = False
            
        print(f"✅ Finished {name} | Processed {total_rows:,} rows in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_kaggle_phase1()
