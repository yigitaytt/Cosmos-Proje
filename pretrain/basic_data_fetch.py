import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def download_dataset(
    dataset_name: str,
    output_path: str,
    format: str = "jsonl",
    cache_dir: str = None
):
    """
    Download dataset from HuggingFace and save locally
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_path: Path to save the dataset
        format: Output format (jsonl, json, parquet, csv)
        cache_dir: Cache directory for HuggingFace datasets
    """
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Output format: {format}")
    
    # Load dataset
    try:
        if cache_dir:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Log available splits
    logger.info(f"Available splits: {list(dataset.keys())}")
    
    # Use train split
    data = dataset['train']
    logger.info(f"Dataset loaded: {len(data)} samples")
    
    # Log dataset info
    logger.info(f"Dataset features: {data.features}")
    if len(data) > 0:
        logger.info(f"First sample: {data[0]}")
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save based on format
    if format == "jsonl":
        save_as_jsonl(data, output_path)
    elif format == "json":
        save_as_json(data, output_path)
    elif format == "parquet":
        save_as_parquet(data, output_path)
    elif format == "csv":
        save_as_csv(data, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Dataset saved to: {output_path}")
    
    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    logger.info(f"File size: {file_size:.2f} MB")
    logger.info(f"Total samples: {len(data)}")


def save_as_jsonl(data, output_path):
    """Save dataset as JSONL (one JSON object per line)"""
    logger.info("Saving as JSONL...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(data, desc="Writing JSONL"):
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')


def save_as_json(data, output_path):
    """Save dataset as single JSON file"""
    logger.info("Saving as JSON...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([sample for sample in data], f, ensure_ascii=False, indent=2)


def save_as_parquet(data, output_path):
    """Save dataset as Parquet file"""
    logger.info("Saving as Parquet...")
    data.to_parquet(output_path)


def save_as_csv(data, output_path):
    """Save dataset as CSV file"""
    logger.info("Saving as CSV...")
    data.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace dataset to local storage")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BILGEM-AI/BILGE-Synthetic-Math",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./bilgem_dataset.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json", "parquet", "csv"],
        default="jsonl",
        help="Output format"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    
    args = parser.parse_args()
    
    download_dataset(
        dataset_name=args.dataset,
        output_path=args.output,
        format=args.format,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()