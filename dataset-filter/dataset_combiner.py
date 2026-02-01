import pandas as pd
import os
import numpy as np

# ================= CONFIGURATION =================
METAMATH_FILE = "turkish_math_FINAL_GOLD.jsonl"
ITU_FILE = "itu_math_FINAL_POLISHED.jsonl"
SYNTHETIC_FILE = "my_synthetic_data.jsonl" #  the data set produced with seeds

# Target counts
COUNTS = {
    "itu": 40000,        # 40k Academic Style
    "metamath": 90000,   # 90k General Logic 
    "synthetic": "all"   # Keep all custom data
}

OUTPUT_FILE = "Final_Turkish_Math_Mix_130k.jsonl"

def main():
    combined_data = []

    # --- 1. LOAD ITU (PLATINUM/ACADEMIC) ---
    print(f"1. Loading ITU ({ITU_FILE})...")
    if os.path.exists(ITU_FILE):
        df_itu = pd.read_json(ITU_FILE, lines=True)
        # Randomly Sample
        if len(df_itu) > COUNTS["itu"]:
            df_itu = df_itu.sample(n=COUNTS["itu"], random_state=42)
        print(f"   -> Added {len(df_itu)} rows from ITU.")
        combined_data.append(df_itu[['instruction', 'output']])
    else:
        print(f"   ! CRITICAL: File not found: {ITU_FILE}")
        return

    # --- 2. LOAD METAMATH (SILVER/VOLUME) ---
    print(f"2. Loading MetaMath ({METAMATH_FILE})...")
    if os.path.exists(METAMATH_FILE):
        df_meta = pd.read_json(METAMATH_FILE, lines=True)
        # Randomly Sample
        if len(df_meta) > COUNTS["metamath"]:
            df_meta = df_meta.sample(n=COUNTS["metamath"], random_state=42)
        print(f"   -> Added {len(df_meta)} rows from MetaMath.")
        combined_data.append(df_meta[['instruction', 'output']])
    else:
        print(f"   ! CRITICAL: File not found: {METAMATH_FILE}")
        return

    # --- 3. LOAD SYNTHETIC ---
    if os.path.exists(SYNTHETIC_FILE):
        print(f"3. Loading Synthetic ({SYNTHETIC_FILE})...")
        df_syn = pd.read_json(SYNTHETIC_FILE, lines=True)
        print(f"   -> Added {len(df_syn)} rows from Synthetic.")
        combined_data.append(df_syn[['instruction', 'output']])

    # --- 4. MERGE & SHUFFLE ---
    print("4. Merging and Shuffling...")
    final_df = pd.concat(combined_data, ignore_index=True)
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   TOTAL ROWS: {len(final_df)}")
    
    # --- 5. SAVE ---
    print(f"5. Saving to {OUTPUT_FILE}...")
    final_df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)
    print("SUCCESS. Ready for Training.")

if __name__ == "__main__":
    main()