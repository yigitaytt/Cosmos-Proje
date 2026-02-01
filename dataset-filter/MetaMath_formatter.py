import pandas as pd
import re
import os
import json

# ================= CONFIGURATION =================
INPUT_FILENAME = "MetaMath_dataset.jsonl"  # Your original dirty file
OUTPUT_FILENAME = "turkish_math_FINAL_GOLD.jsonl" # The perfect output

# 1. KILL WORDS (If these exist, DROP the row)
FORBIDDEN_TERMS = [
    r'\bfeet\b', r'\bfoot\b', r'\bft\b',
    r'\bmile\b', r'\bmiles\b', r'\bmil\b', 
    r'\binch\b', r'\binches\b', 
    r'\byard\b', r'\byards\b',
    r'\bpound\b', r'\bpounds\b', r'\blb\b', r'\blbs\b',
    r'\bunce\b', r'\bounces\b', r'\boz\b',
    r'\bgallon\b', r'\bgal\b',
    r'\bacre\b', r'\bacres\b',
    r'\bfahrenheit\b',
    r'\bmph\b',
    r'\bdolar\b', r'\beuro\b', 
    r'\bcent\b', r'\bpenny\b', r'\bpence\b'
]

# 2. CURRENCY SYMBOL PATTERN
# Catches "$5" OR "5$"
CURRENCY_PATTERN = r'(\$(?=\d)|(?<=\d)\$)'
# =================================================

def repair_text(text):
    if not isinstance(text, str): return text
    # Fix 1: Unescape slashes (400\/4 -> 400/4)
    text = text.replace(r"\/", "/")
    # Fix 2: Remove broken "elde ederiz" from LaTeX
    text = re.sub(r'=\\dfrac\{\s*elde ederiz\.\s*', r'=\\dfrac{', text, flags=re.IGNORECASE)
    text = re.sub(r'=\\frac\{\s*elde ederiz\.\s*', r'=\\frac{', text, flags=re.IGNORECASE)
    # Fix 3: Clean spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_row_clean(text):
    if not isinstance(text, str): return False
    text_lower = text.lower()
    
    # Check 1: Forbidden Words
    for pattern in FORBIDDEN_TERMS:
        if re.search(pattern, text_lower):
            return False 
            
    # Check 2: Currency Symbols ($5 or 5$)
    if re.search(CURRENCY_PATTERN, text):
        return False 
        
    return True 

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILENAME)
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)

    print(f"--- MASTER PROCESSOR STARTED ---")
    print(f"Reading from: {INPUT_FILENAME}")

    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {INPUT_FILENAME}")
        return

    # 1. LOAD DATA
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} original rows.")

    # 2. STANDARDIZE COLUMNS
    if 'query' in df.columns:
        df = df.rename(columns={'query': 'instruction', 'response': 'output'})
    elif 'question' in df.columns:
        df = df.rename(columns={'question': 'instruction', 'answer': 'output'})
    
    # 3. REPAIR PHASE (Fixing broken text)
    print("Step 1: Repairing text (fixing slashes & LaTeX)...")
    df['instruction'] = df['instruction'].apply(repair_text)
    df['output'] = df['output'].apply(repair_text)

    # 4. FILTER PHASE (Dropping bad rows)
    print("Step 2: Filtering bad rows (Imperial units & Dollars)...")
    
    # Check for the specific row you mentioned (25$)
    # We use a mask to find it before filtering to prove it exists
    bad_row_check = df['output'].str.contains(r'25\$', regex=True)
    print(f"   -> Found {bad_row_check.sum()} rows with '25$' before filtering.")

    mask_instr = df['instruction'].apply(is_row_clean)
    mask_out = df['output'].apply(is_row_clean)
    
    df_clean = df[mask_instr & mask_out].copy()
    
    dropped = len(df) - len(df_clean)
    print(f"   -> DROPPED {dropped} rows.")
    print(f"   -> Remaining: {len(df_clean)} rows.")

    # 5. SAVE
    print(f"Step 3: Saving to {OUTPUT_FILENAME}...")
    df_clean.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print("Done! Use THIS file for training.")

if __name__ == "__main__":
    main()