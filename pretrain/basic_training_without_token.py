import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import tempfile
import csv
from datetime import datetime
import torch
import re
import numpy as np 
import shutil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback,
)
from datasets import load_dataset, Dataset

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def get_env_variable(var_name: str, default=None, var_type=str):
    """Get environment variable with type conversion"""
    value = os.environ.get(var_name, default)
    if value is None:
        return None
    
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes')
    elif var_type == int:
        return int(value)
    elif var_type == float:
        return float(value)
    else:
        return value

class LocalLoggingCallback(TrainerCallback):
    """
    Save all training metrics locally to CSV files.
    No external dependencies, everything stored on disk.
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate CSV files
        self.train_log_file = self.output_dir / "train_metrics.csv"
        self.eval_log_file = self.output_dir / "eval_metrics.csv"
        self.summary_file = self.output_dir / "training_summary.txt"
        
        # Initialize train metrics CSV
        with open(self.train_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'step', 'epoch', 'loss', 'train_perplexity', 
                'learning_rate', 'grad_norm'
            ])
        
        # Initialize eval metrics CSV
        with open(self.eval_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'step', 'epoch', 'eval_loss', 'eval_perplexity',
                'eval_runtime', 'eval_samples_per_second'
            ])
        
        logger.info(f"Local logging initialized:")
        logger.info(f"  Train metrics: {self.train_log_file}")
        logger.info(f"  Eval metrics: {self.eval_log_file}")
        logger.info(f"  Summary: {self.summary_file}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics"""
        if logs is None:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Training metrics
        if 'loss' in logs:
            train_ppl = np.exp(min(logs['loss'], 100)) 
            
            with open(self.train_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    state.global_step,
                    logs.get('epoch', ''),
                    logs.get('loss', ''),
                    train_ppl,
                    logs.get('learning_rate', ''),
                    logs.get('grad_norm', ''),
                ])
            
            # Console output
            logger.info(
                f"Step {state.global_step:6d} | "
                f"Loss: {logs['loss']:.4f} | "
                f"PPL: {train_ppl:.4f} | "
                f"LR: {logs.get('learning_rate', 0):.2e}"
            )
        
        # Evaluation metrics
        if 'eval_loss' in logs:
            eval_ppl = np.exp(logs['eval_loss'])
            
            with open(self.eval_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    state.global_step,
                    logs.get('epoch', ''),
                    logs.get('eval_loss', ''),
                    eval_ppl,
                    logs.get('eval_runtime', ''),
                    logs.get('eval_samples_per_second', ''),
                ])
            
            # Console output
            logger.info("=" * 80)
            logger.info(f"EVALUATION at Step {state.global_step}")
            logger.info(f"  Eval Loss: {logs['eval_loss']:.4f}")
            logger.info(f"  Eval Perplexity: {eval_ppl:.4f}")
            logger.info(f"  Runtime: {logs.get('eval_runtime', 0):.2f}s")
            logger.info("=" * 80)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log training start"""
        with open(self.summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING STARTED\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write("\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log training end with summary"""
        with open(self.summary_file, 'a') as f:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("TRAINING COMPLETED\n")
            f.write("=" * 80 + "\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total steps: {state.global_step}\n")
            f.write(f"Total epochs: {state.epoch}\n")
            
            # Read final metrics
            try:
                train_df = pd.read_csv(self.train_log_file)
                eval_df = pd.read_csv(self.eval_log_file)
                
                f.write("\nFinal Metrics:\n")
                if len(train_df) > 0:
                    f.write(f"  Final Train Loss: {train_df['loss'].iloc[-1]:.4f}\n")
                    f.write(f"  Final Train Perplexity: {train_df['train_perplexity'].iloc[-1]:.4f}\n")
                if len(eval_df) > 0:
                    f.write(f"  Final Eval Loss: {eval_df['eval_loss'].iloc[-1]:.4f}\n")
                    f.write(f"  Final Eval Perplexity: {eval_df['eval_perplexity'].iloc[-1]:.4f}\n")
                    f.write(f"  Best Eval Perplexity: {eval_df['eval_perplexity'].min():.4f}\n")
            except Exception as e:
                error_msg = f"\nCould not compute final metrics: {e}\n"
                f.write(error_msg)
                logger.warning(error_msg)
        
        logger.info(f"Training summary saved to: {self.summary_file}")
        logger.info("=" * 80)

class TrainingConfig:
    """Configuration class for pretraining"""
    def __init__(self):
        # Workspace and paths
        self.workspace = get_env_variable("TRAINING_WORKSPACE", os.getcwd())
        self.job_id = get_env_variable("JOB_ID", "default")
        self.output_dir = get_env_variable("OUTPUT_DIR", "./output")
        self.cache_dir = get_env_variable("CACHE_DIR", "./cache")
        
        # Model and dataset (LOCAL PATHS)
        self.model_path = get_env_variable("MODEL_PATH", "./turkish-gpt2-medium")
        self.dataset_path = get_env_variable("DATASET_PATH", "./bilgem_dataset.jsonl")
        
        # Dataset configuration
        self.max_length = get_env_variable("MAX_LENGTH", "4096", int)
        self.eval_split_ratio = get_env_variable("EVAL_SPLIT_RATIO", "0.001", float)  # 5% for eval
        
        # Training hyperparameters
        self.num_epochs = get_env_variable("NUM_EPOCHS", "3", int)
        self.learning_rate = get_env_variable("LEARNING_RATE", "1e-4", float)
        self.warmup_steps = get_env_variable("WARMUP_STEPS", "2000", int)
        self.weight_decay = get_env_variable("WEIGHT_DECAY", "0.01", float)
        
        # Batch size and gradient accumulation
        self.batch_size = get_env_variable("BATCH_SIZE", "16", int)
        self.gradient_accum = get_env_variable("GRADIENT_ACCUM", "8", int)
        self.dataloader_num_workers = get_env_variable("DATALOADER_NUM_WORKERS", "8", int)
        
        # Mixed precision
        self.use_bf16 = get_env_variable("USE_BF16", "true", bool)
        
        # Logging and checkpointing
        self.logging_steps = get_env_variable("LOGGING_STEPS", "10", int)
        self.eval_steps = get_env_variable("EVAL_STEPS", "500", int)
        self.save_steps = get_env_variable("SAVE_STEPS", "500", int)
        self.save_total_limit = get_env_variable("SAVE_TOTAL_LIMIT", "2", int)
        
        # Evaluation strategy
        self.eval_strategy = get_env_variable("EVAL_STRATEGY", "steps")  # steps or epoch
        self.load_best_model = get_env_variable("LOAD_BEST_MODEL", "true", bool)
        self.metric_for_best_model = get_env_variable("METRIC_FOR_BEST_MODEL", "loss")

        self.lr_scheduler_type = get_env_variable("LR_SCHEDULER_TYPE", "cosine")
        
        self.max_grad_norm = get_env_variable("MAX_GRAD_NORM", "1.0", float)

        # Resume training
        self.resume_from_checkpoint = get_env_variable("RESUME_FROM_CHECKPOINT", "auto")
        if self.resume_from_checkpoint == "false":
            self.resume_from_checkpoint = None
        elif self.resume_from_checkpoint == "auto":
            self.resume_from_checkpoint = True
        
        # Random seed
        self.seed = get_env_variable("SEED", "42", int)
    
    def validate_paths(self):
        """Validate that model and dataset paths exist"""
        model_path = Path(self.model_path)
        dataset_path = Path(self.dataset_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        logger.info(f"✓ Model path validated: {self.model_path}")
        logger.info(f"✓ Dataset path validated: {self.dataset_path}")
    
    def log_config(self):
        """Log all configuration parameters"""
        logger.info("=" * 80)
        logger.info("PRETRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Job ID: {self.job_id}")
        logger.info(f"Workspace: {self.workspace}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        logger.info("-" * 80)
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Dataset Path: {self.dataset_path}")
        logger.info(f"Max Sequence Length: {self.max_length}")
        logger.info(f"Eval Split Ratio: {self.eval_split_ratio}")
        logger.info("-" * 80)
        logger.info(f"Number of Epochs: {self.num_epochs}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Weight Decay: {self.weight_decay}")
        logger.info(f"Warmup Steps: {self.warmup_steps}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Gradient Accumulation Steps: {self.gradient_accum}")
        logger.info(f"Effective Batch Size: {self.batch_size * self.gradient_accum}")
        logger.info(f"Use BF16: {self.use_bf16}")
        logger.info("-" * 80)
        logger.info(f"Logging Steps: {self.logging_steps}")
        logger.info(f"Eval Steps: {self.eval_steps}")
        logger.info(f"Save Steps: {self.save_steps}")
        logger.info(f"Save Total Limit: {self.save_total_limit}")
        logger.info(f"Eval Strategy: {self.eval_strategy}")
        logger.info(f"Load Best Model: {self.load_best_model}")
        logger.info(f"Metric for Best Model: {self.metric_for_best_model}")
        logger.info(f"LR Scheduler Type: {self.lr_scheduler_type}")
        logger.info(f"Max Grad Norm: {self.max_grad_norm}")
        logger.info(f"Resume from Checkpoint: {self.resume_from_checkpoint}")
        logger.info(f"Random Seed: {self.seed}")
        logger.info("=" * 80)

def prepare_pretraining_dataset(tokenizer, dataset_path, max_length, eval_split_ratio, seed):
    """Load and tokenize the local JSONL dataset for pretraining"""
    # Load JSONL file
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Log first sample for verification
    if len(dataset) > 0:
        logger.info("First sample from dataset:")
        logger.info(f"Keys: {list(dataset[0].keys())}")
        logger.info(f"Sample: {dataset[0]}")
    
    def prepare_text(examples):
        texts = []
        
        if 'text' in examples:
            # Already prepared text
            texts = examples['text']
        else:
            raise ValueError(f"Unknown dataset format. Available keys: {examples.keys()}")
        
        return {'text': texts}
    
    # First, prepare the text
    logger.info("Preparing text for pretraining...")
    dataset = dataset.map(
        prepare_text,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != 'text'],
        desc="Preparing text"
    )
    
    # Split into train and eval
    logger.info(f"Splitting dataset: {1-eval_split_ratio:.1%} train, {eval_split_ratio:.1%} eval")
    split_dataset = dataset.train_test_split(
        test_size=eval_split_ratio,
        seed=seed,
        shuffle=True
    )
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    def tokenize_function(examples):
        # Tokenize with room for EOS token
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length - 1,  # Reserve space for EOS
            padding=False,
            return_tensors=None,
        )
        
        # Add EOS token to each sequence
        outputs["input_ids"] = [
            ids + [tokenizer.eos_token_id] 
            for ids in outputs["input_ids"]
        ]
        
        # Add attention mask for EOS token if present
        if "attention_mask" in outputs:
            outputs["attention_mask"] = [
                mask + [1]
                for mask in outputs["attention_mask"]
            ]
        
        return outputs
    
    logger.info("Tokenizing train dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        num_proc=8,
    )
    
    logger.info("Tokenizing eval dataset...")
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
        num_proc=8,
    )

    logger.info("Verifying EOS token placement...")

    num_samples_to_check = min(100, len(tokenized_train))
    correct_eos_placement = 0

    for i in range(num_samples_to_check):
        sample_ids = tokenized_train[i]['input_ids']
        
        # Find where EOS token is
        try:
            eos_idx = sample_ids.index(tokenizer.eos_token_id)
            
            # Check if there are any non-pad tokens after EOS
            tokens_after_eos = sample_ids[eos_idx + 1:]
            
            # All tokens after EOS should be padding (or nothing)
            if len(tokens_after_eos) == 0:
                # EOS is last token (no padding) - CORRECT
                correct_eos_placement += 1
            elif all(token == tokenizer.pad_token_id for token in tokens_after_eos):
                # EOS followed only by padding - CORRECT
                correct_eos_placement += 1
            else:
                # There are non-pad tokens after EOS - WRONG!
                logger.error(f"Sample {i}: Non-pad tokens after EOS!")
                logger.error(f"  Tokens after EOS: {tokens_after_eos[:5]}")
                
        except ValueError:
            # No EOS token found
            logger.error(f"Sample {i}: No EOS token found!")
            logger.error(f"  Last 5 tokens: {sample_ids[-5:]}")

    success_rate = correct_eos_placement / num_samples_to_check

    logger.info(f"Checked {num_samples_to_check} samples:")
    logger.info(f"  Correct EOS placement: {correct_eos_placement}/{num_samples_to_check} ({success_rate:.1%})")

    if success_rate < 1.0:
        logger.error("ERROR: EOS tokens are not correctly placed before padding!")
        raise ValueError("EOS token placement is incorrect")
    else:
        logger.info("✓ All EOS tokens correctly placed before padding")
    
    logger.info(f"Train dataset tokenization complete: {len(tokenized_train)} samples")
    logger.info(f"Eval dataset tokenization complete: {len(tokenized_eval)} samples")
    
    def count_tokens(dataset):
        total = sum(len(sample['input_ids']) for sample in dataset)
        return total

    train_tokens = count_tokens(tokenized_train)
    eval_tokens = count_tokens(tokenized_eval)

    avg_train_tokens = train_tokens / len(tokenized_train)
    avg_eval_tokens = eval_tokens / len(tokenized_eval)
    
    logger.info(f"Train tokens: {train_tokens:,} (avg: {avg_train_tokens:.2f} per sample)")
    logger.info(f"Eval tokens: {eval_tokens:,} (avg: {avg_eval_tokens:.2f} per sample)")
    logger.info(f"Total tokens: {train_tokens + eval_tokens:,}")
    
    return tokenized_train, tokenized_eval


def main():
    """Main pretraining function"""
    # Load configuration
    config = TrainingConfig()
    
    # Validate paths
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Log configuration
    config.log_config()
    
    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {config.output_dir}")
    
    # Load tokenizer from local path
    logger.info(f"Loading tokenizer from {config.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    tokenizer.padding_side = 'right'

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token for causal LM training")
    
    logger.info(f"Tokenizer loaded - Vocab size: {len(tokenizer)}")
    
    # Load model from local path
    logger.info(f"Loading model from {config.model_path}")
    try:
        # Determine dtype based on BF16 setting
        if config.use_bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using BFloat16 precision")
        elif torch.cuda.is_available():
            dtype = torch.float16
            logger.info("Using Float16 precision")
        else:
            dtype = torch.float32
            logger.info("Using Float32 precision")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=dtype,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    logger.info(f"Model loaded: {model.num_parameters():,} parameters")
    
    # ONLY pad token setup - no other tokenizer modifications
    if tokenizer.pad_token is None:
        logger.info("Adding special [PAD] token")
        if '[PAD]' in tokenizer.get_vocab():
            pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
            tokenizer.pad_token = '[PAD]'
            tokenizer.pad_token_id = pad_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            
    logger.info(f"Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"Final Vocab Size: {len(tokenizer)}")
    
    # Prepare dataset for pretraining with train/eval split
    try:
        tokenized_train, tokenized_eval = prepare_pretraining_dataset(
            tokenizer,
            config.dataset_path,
            config.max_length,
            config.eval_split_ratio,
            config.seed
        )
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise
    
    # Data collator for causal language modeling (pretraining)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM for pretraining
        return_tensors="pt",
    )

    logger.info("Verifying data collator padding behavior...")
    test_batch = data_collator([tokenized_train[0], tokenized_train[1]])
    if tokenizer.pad_token_id in test_batch['input_ids']:
        pad_positions = (test_batch['input_ids'] == tokenizer.pad_token_id)
        labels_at_pad = test_batch['labels'][pad_positions]
        if not torch.all(labels_at_pad == -100):
            logger.warning("WARNING: Padding tokens are NOT being masked in labels!")
        else:
            logger.info("✓ Padding tokens correctly masked in labels")
    
    # Determine precision settings
    use_fp16 = torch.cuda.is_available() and not config.use_bf16
    use_bf16 = config.use_bf16 and torch.cuda.is_bf16_supported()
    
    # Training arguments optimized for pretraining
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accum,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        
        # Evaluation settings
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        
        # Logging settings
        logging_steps=config.logging_steps,
        logging_first_step=True,
        logging_strategy="steps",
        
        # Saving settings
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.eval_strategy,  # Align with eval_strategy
        
        # Best model settings
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,  # Lower loss is better
        
        lr_scheduler_type=config.lr_scheduler_type,  
        max_grad_norm=config.max_grad_norm,
        # Precision settings
        fp16=use_fp16,
        bf16=use_bf16,
        
        # Data settings
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        
        disable_tqdm=True,

        # Reporting
        report_to=[],
        logging_dir=f"{config.output_dir}/logs",
        
        # Other settings
        ddp_find_unused_parameters=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    
    model = torch.compile(model)

    # Initialize trainer
    logger.info("Initializing Trainer for Pretraining")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[LocalLoggingCallback(config.output_dir)], 
    )
    
    logger.info("⚡ RUNNING SMOKE TEST (Safety Check)...")
    try:
        # 1. Test Evaluation (Runs on a small subset automatically)
        logger.info("   1. Testing Evaluation Loop...")
        metrics = trainer.evaluate()
        logger.info(f"      ✓ Evaluation passed! Metrics: {metrics}")

        # 2. Test Saving (Saves to a temporary folder)
        logger.info("   2. Testing Save Logic...")
        test_save_path = os.path.join(config.output_dir, "test_checkpoint")
        trainer.save_model(test_save_path)
        tokenizer.save_pretrained(test_save_path)
        
        # Verify the file actually exists
        if os.path.exists(os.path.join(test_save_path, "config.json")):
            logger.info(f"      ✓ Model saved successfully to {test_save_path}")
        else:
            raise FileNotFoundError("Save reported success but config.json is missing!")
            
        # Clean up (Optional: remove the test folder)
        shutil.rmtree(test_save_path)
        logger.info("      ✓ Test checkpoint cleaned up.")

        logger.info("✅ SMOKE TEST PASSED. Starting full training now...")
    except Exception as e:  # <--- ADD THIS BLOCK
        logger.error(f"❌ SMOKE TEST FAILED: {e}")
        logger.error("Fix the error above before starting the 10-hour run!")
        sys.exit(1)

    # Print GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Calculate training statistics
    total_steps = (len(tokenized_train) // (config.batch_size * config.gradient_accum)) * config.num_epochs
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Warmup steps: {config.warmup_steps}")
    logger.info(f"Warmup ratio: {config.warmup_steps / total_steps:.2%}")
    logger.info(f"Evaluation will run every {config.eval_steps} steps")
    
    # Train
    logger.info("=" * 80)
    logger.info("STARTING PRETRAINING")
    logger.info("=" * 80)
    
    try:
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint if config.resume_from_checkpoint else None)
    except Exception as e:
        logger.error(f"Pretraining failed with error: {e}", exc_info=True)
        raise
    
    # Save final model
    final_model_path = f"{config.output_dir}/final_model_without"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # If load_best_model_at_end is True, also save the best model separately
    if config.load_best_model:
        best_model_path = f"{config.output_dir}/best_model"
        logger.info(f"Saving best model to {best_model_path}")
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
    
    logger.info("=" * 80)
    logger.info("PRETRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Final model saved to: {final_model_path}")
    if config.load_best_model:
        logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Logs saved to: {config.output_dir}/logs")
    logger.info(f"Job ID: {config.job_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pretraining script failed: {e}", exc_info=True)
        sys.exit(1)