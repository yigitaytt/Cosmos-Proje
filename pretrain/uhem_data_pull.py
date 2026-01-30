import os
import logging
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

DATASET_NAME = "BILGEM-AI/BILGE-Synthetic-Math"
MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-medium"
MAX_LEN = 1024

# Shared workspace for all jobs
WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', ''))

# Dataset gets a fixed, shared location (not job-specific)
DATASET_OUTPUT_DIR = WORKSPACE / 'data' / 'prepared' / 'tokenized_math_data'

# Create directories
DATASET_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATASET_OUTPUT_DIR.parent / 'prepare_dataset.log'),
        logging.StreamHandler()
    ]
)

def tokenize_function(examples, tokenizer):
    """Tokenize the 'text' column directly"""
    texts = examples['text']
    processed_texts = [t + tokenizer.eos_token for t in texts]
    
    return tokenizer(
        processed_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

def validate_tokenization(dataset, sample_size=100):
    sample_size = min(sample_size, len(dataset))
    sample = dataset.select(range(sample_size))
    
    max_len = max(len(x) for x in sample['input_ids'])
    avg_len = sum(len(x) for x in sample['input_ids']) / len(sample)
    
    logging.info(f"Max token length in sample: {max_len}")
    logging.info(f"Average token length: {avg_len:.1f}")
    
    has_required = all(key in dataset.column_names for key in ['input_ids', 'attention_mask'])
    if not has_required:
        raise ValueError("Tokenization failed: missing required columns")
    
    return True

def main():
    try:
        # Check if dataset already exists
        if DATASET_OUTPUT_DIR.exists():
            logging.warning(f"Dataset already exists at {DATASET_OUTPUT_DIR}")
        
        logging.info(f"Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        logging.info(f"Downloading dataset: {DATASET_NAME}")
        raw_dataset = load_dataset(DATASET_NAME, split="train")
        
        logging.info(f"Dataset loaded: {len(raw_dataset)} rows")
        logging.info(f"Columns: {raw_dataset.column_names}")
        
        # Verify 'text' column exists
        if 'text' not in raw_dataset.column_names:
            raise ValueError(f"'text' column not found. Available columns: {raw_dataset.column_names}")
        
        logging.info(f"Using 'text' column")
        logging.info(f"Sample text (first 200 chars): {raw_dataset[0]['text'][:200]}")
        
        logging.info("Starting tokenization...")
        tokenized_dataset = raw_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            num_proc=os.cpu_count()//2,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing"
        )
        
        logging.info("Validating tokenization...")
        validate_tokenization(tokenized_dataset)
        
        logging.info(f"Saving to disk: {DATASET_OUTPUT_DIR}")
        tokenized_dataset.save_to_disk(str(DATASET_OUTPUT_DIR))
        
        # Calculate and log dataset size
        dataset_size_gb = sum(
            f.stat().st_size for f in DATASET_OUTPUT_DIR.rglob('*') if f.is_file()
        ) / 1e9
        
        logging.info(f"Successfully saved {len(tokenized_dataset)} samples ({dataset_size_gb:.2f} GB)")
        logging.info(f"Dataset location: {DATASET_OUTPUT_DIR}")
        logging.info("Preparation complete!")
        
        # Write metadata file for training script to verify
        metadata = {
            'dataset_name': DATASET_NAME,
            'model_name': MODEL_NAME,
            'num_samples': len(tokenized_dataset),
            'max_length': MAX_LEN,
            'size_gb': dataset_size_gb,
            'text_column': 'text'
        }
        
        with open(DATASET_OUTPUT_DIR.parent / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        logging.error(f"Preparation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()