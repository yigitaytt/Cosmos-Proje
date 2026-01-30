import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
import shutil

model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

def main():
    dataset_path = "big_data/shard_0"
    print(f"Loading dataset from: {dataset_path}")

    # 1. Load Data
    raw_dataset = load_from_disk(dataset_path)

    # 2. CREATE PROXY DATASET (Crucial Step)
    # We select only the first 2,000 examples to make this fast.
    # If your dataset is shuffled, taking the first N is fine. If not, use .shuffle() first.
    proxy_dataset = raw_dataset.select(range(4000))

    print(f"Proxy dataset created with {len(proxy_dataset)} samples.")

    # 3. Tokenize
    column_names = raw_dataset.column_names
    tokenized_datasets = proxy_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=4
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Define the Sweep
    learning_rates_to_test = [1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 1e-5]
    results = {}

    print(f"\nStarting Sweep on {torch.cuda.device_count()} GPU(s)...")
    print("-" * 50)

    for lr in learning_rates_to_test:
        print(f"\n>>> Testing Learning Rate: {lr}")

        # Reload model fresh for each run to avoid weight contamination
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.config.pad_token_id = tokenizer.eos_token_id

        run_name = f"sweep_lr_{lr}"

        # 5. Fast Training Arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{run_name}",
            overwrite_output_dir=True,
            # We use max_steps instead of epochs for precise speed control
            max_steps=30,              # Run for just 100 steps
            per_device_train_batch_size=2,
            gradient_accumulation_steps=64,
            learning_rate=lr,           # The variable we are testing
            weight_decay=0.1,
            fp16=True,
            logging_steps=10,
            save_strategy="no",         # Don't save checkpoints to save disk space
            report_to="none",
            ddp_find_unused_parameters=False,
            # lr_scheduler_type="cosine" # Remember to add this back when doing the actual training
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets,
        )

        trainer.train()

        # Capture the final training loss
        final_loss = trainer.state.log_history[-1].get('train_loss')
        # Sometimes the last log is an 'epoch' stat, so we grab the last 'loss' entry safely
        if final_loss is None:
            for log in reversed(trainer.state.log_history):
                if 'loss' in log:
                    final_loss = log['loss']
                    break

        results[lr] = final_loss
        print(f">>> Finished LR {lr} with Final Loss: {final_loss}")

        # Cleanup to save memory
        del model
        del trainer
        torch.cuda.empty_cache()

    # 6. Print Summary
    print("\n" + "="*30)
    print("SWEEP RESULTS (Lower Loss is Better)")
    print("="*30)
    sorted_results = sorted(results.items(), key=lambda x: x[1] if x[1] is not None else float('inf'))

    for lr, loss in sorted_results:
        print(f"LR: {lr:.1e} | Loss: {loss:.4f}")

    winner = sorted_results[0][0]
    print("="*30)
    print(f"RECOMMENDED LEARNING RATE: {winner:.1e}")
    print("="*30)

if __name__ == "__main__":
    main()