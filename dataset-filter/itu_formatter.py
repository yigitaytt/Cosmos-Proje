import pandas as pd
import re
import os
import json

# INPUT: The file you just showed me
INPUT_FILENAME = "itu_math_cleaned.jsonl" 
# OUTPUT: The truly ready file
OUTPUT_FILENAME = "itu_math_FINAL_POLISHED.jsonl"

def polished_clean(text):
    if not isinstance(text, str): return ""
    
    # 1. REMOVE IMAGE LINKS (Strict)
    # Catches ![](...) and typical url patterns
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'https?://\S+', '', text)

    # 2. REMOVE "GHOST" REFERENCES
    # Removes lines saying "The figure below shows..."
    text = re.sub(r'(Aşağıdaki|Yandaki) (şekil|resim|grafik).*', '', text, flags=re.IGNORECASE)

    # 3. REMOVE META-COMMENTARY
    # Removes "Açıklama", "Not", "Öğrenciler" at the start of paragraphs
    lines = text.split('\n')
    valid_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Açıklama") or "Öğrenciler ayrıca" in stripped:
            continue
        if stripped.startswith("Not.") or stripped.startswith("PSC"):
            continue
        valid_lines.append(line)
    text = "\n".join(valid_lines)

    # 4. THE "GLUE" FIX (CRITICAL FOR TOKENIZATION)
    # Adds a space before $ if there isn't one
    text = re.sub(r'(?<=\S)\$', ' $', text)
    # Adds a space after $ if there isn't one (and it's followed by text)
    text = re.sub(r'\$(?=\S)', '$ ', text)
    
    # 5. Collapse excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def main():
    print(f"Reading {INPUT_FILENAME}...")
    
    # Load the JSONL
    data = []
    with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows.")

    print("Polishing text (Fixing spacing and removing artifacts)...")
    
    # Apply to BOTH columns
    df['instruction'] = df['instruction'].apply(polished_clean)
    df['output'] = df['output'].apply(polished_clean)

    # Final sanity check: Drop empty rows
    df = df[df['instruction'].str.len() > 5]
    df = df[df['output'].str.len() > 5]

    print(f"Saving {len(df)} polished rows to {OUTPUT_FILENAME}...")
    df.to_json(OUTPUT_FILENAME, orient='records', lines=True, force_ascii=False)
    print("Done. Use THIS file for the merger.")

if __name__ == "__main__":
    main()