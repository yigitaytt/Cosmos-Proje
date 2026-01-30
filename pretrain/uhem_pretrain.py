import os
import logging
import torch
import signal
import sys
import json
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from pathlib import Path

MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-medium"

# Shared workspace
WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', ''))
if not WORKSPACE or not WORKSPACE.exists():
    raise ValueError("TRAINING_WORKSPACE must be set and must exist")

# Dataset location (prepared separately, shared across all training runs)
DATASET_PATH = WORKSPACE / 'data' / 'prepared' / 'tokenized_math_data'

# Training output is job-specific
JOB_ID = os.getenv('SLURM_JOB_ID') or os.getenv('JOB_ID') or 'local'
OUTPUT_DIR = WORKSPACE / 'models' / JOB_ID / 'turkish-gpt2-math-final'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)

def get_optimal_batch_size():
    if not torch.cuda.is_available():
        return 1, 128
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logging.info(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb >= 40:
        return 16, 8
    elif gpu_memory_gb >= 24:
        return 8, 16
    elif gpu_memory_gb >= 16:
        return 4, 32
    else:
        return 2, 64

def log_gpu_usage():
    if not torch.cuda.is_available():
        return
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9
        logging.info(f"GPU {i}: {allocated:.1f}/{total:.1f} GB allocated, {reserved:.1f} GB reserved")

def verify_dataset(dataset_path):
    """Verify dataset exists and is valid"""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please run prepare_dataset.py first to create the dataset."
        )
    
    # Check for required files
    required_files = ['dataset_info.json', 'state.json']
    missing_files = [f for f in required_files if not (dataset_path / f).exists()]
    if missing_files:
        raise ValueError(
            f"Dataset at {dataset_path} appears corrupted. Missing files: {missing_files}\n"
            f"Please re-run prepare_dataset.py"
        )
    
    # Log metadata if available
    metadata_path = dataset_path.parent / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logging.info("Dataset metadata:")
        for key, value in metadata.items():
            logging.info(f"  {key}: {value}")
    
    # Calculate dataset size
    dataset_size_gb = sum(
        f.stat().st_size for f in dataset_path.rglob('*') if f.is_file()
    ) / 1e9
    
    logging.info(f"Dataset size: {dataset_size_gb:.2f} GB")
    
    if dataset_size_gb > 100:
        logging.warning("Very large dataset detected. Consider using streaming or sharding.")
    
def check_for_checkpoints(output_dir):
    if not output_dir.exists():
        return False, None
    
    checkpoints = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith("checkpoint-")]
    if not checkpoints:
        return False, None
    
    checkpoint_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.name.split("-")[1])
            checkpoint_steps.append((step, ckpt))
        except (IndexError, ValueError):
            continue
    
    if not checkpoint_steps:
        return False, None
    
    latest_step, latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])
    logging.info(f"Found checkpoint at step {latest_step}")
    return True, latest_checkpoint

class SignalHandler:
    def __init__(self, trainer, output_dir):
        self.trainer = trainer
        self.output_dir = output_dir
        self.received_signal = False
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        if not self.received_signal:
            self.received_signal = True
            logging.warning(f"Received signal {signum}. Saving checkpoint immediately...")
            save_path = self.output_dir / "timeout_checkpoint"
            self.trainer.save_model(str(save_path))
            self.trainer.save_state()
            logging.info(f"Checkpoint saved to {save_path}. Exiting.")
            sys.exit(0)

def main():
    try:
        logging.info(f"Job ID: {JOB_ID}")
        logging.info(f"Workspace: {WORKSPACE}")
        logging.info(f"Dataset path: {DATASET_PATH}")
        logging.info(f"Output directory: {OUTPUT_DIR}")
        
        # Verify dataset exists and is valid
        logging.info("Verifying dataset...")
        verify_dataset(DATASET_PATH)
        
        logging.info(f"Loading tokenizer: {MODEL_NAME}")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        logging.info(f"Loading preprocessed data from {DATASET_PATH}")
        full_dataset = load_from_disk(str(DATASET_PATH))
        
        logging.info("Creating train/validation split (99%/1%)")
        split_datasets = full_dataset.train_test_split(test_size=0.01, seed=42)
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]
        
        logging.info(f"Training samples: {len(train_dataset):,}")
        logging.info(f"Validation samples: {len(eval_dataset):,}")
        logging.info(f"GPUs available: {torch.cuda.device_count()}")
        
        log_gpu_usage()
        
        has_checkpoint, checkpoint_path = check_for_checkpoints(OUTPUT_DIR)
        
        if has_checkpoint:
            logging.info(f"Found existing checkpoint: {checkpoint_path}")
            logging.info("Resuming training from checkpoint...")
            should_resume = True
            should_overwrite = False
        else:
            logging.info("No checkpoints found. Starting fresh training...")
            should_resume = False
            should_overwrite = True
        
        logging.info("Loading model...")
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        model.config.pad_token_id = tokenizer.eos_token_id
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        batch_size, grad_accum = get_optimal_batch_size()
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            effective_batch = batch_size * grad_accum * num_gpus
            logging.info(f"Batch config: {batch_size} per device × {grad_accum} accum × {num_gpus} GPUs = {effective_batch} effective")
        else:
            effective_batch = batch_size * grad_accum
            logging.info(f"Batch config: {batch_size} per device × {grad_accum} accum (CPU) = {effective_batch} effective")
        
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            overwrite_output_dir=should_overwrite,
            num_train_epochs=1,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=500,
            weight_decay=0.1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=grad_accum,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="none",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            disable_tqdm=False,  # Enable progress bar for monitoring
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        handler = SignalHandler(trainer, OUTPUT_DIR)

        logging.info("Starting training...")
        log_gpu_usage()
        
        if should_resume and checkpoint_path:
            trainer.train(resume_from_checkpoint=str(checkpoint_path))
        else:
            trainer.train()
        
        logging.info("Training completed. Saving final model...")
        final_model_path = OUTPUT_DIR / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        logging.info(f"Model saved to {final_model_path}")
        logging.info("All done!")
        
    except FileNotFoundError as e:
        logging.error(str(e))
        logging.error("Please run prepare_dataset.py before training.")
        sys.exit(1)
        
    except torch.cuda.OutOfMemoryError:
        logging.error("GPU out of memory! Reduce batch size in get_optimal_batch_size()")
        if 'trainer' in locals():
            emergency_path = OUTPUT_DIR / "emergency_checkpoint"
            logging.info(f"Saving emergency checkpoint to {emergency_path}")
            trainer.save_model(str(emergency_path))
        raise
        
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        if 'trainer' in locals():
            interrupt_path = OUTPUT_DIR / "interrupted_checkpoint"
            logging.info(f"Saving interrupted checkpoint to {interrupt_path}")
            trainer.save_model(str(interrupt_path))
        raise
        
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        if 'trainer' in locals():
            error_path = OUTPUT_DIR / "error_checkpoint"
            logging.info(f"Attempting to save checkpoint to {error_path}")
            try:
                trainer.save_model(str(error_path))
            except:
                logging.error("Could not save error checkpoint")
        raise


if __name__ == "__main__":
    main()