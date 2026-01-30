import torch
from transformers import (
    AutoTokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

# 1. PREPARE THE DATA (Arrow Format)
# ---------------------------------------------------------
# Path to the folder containing the arrow files
dataset_path = r"C:\Users\pc\OneDrive - Yildiz Technical University\Desktop\Cosmos\data\shard_0"

print(f"Loading dataset from: {dataset_path}")

# Load the arrow dataset directly from disk
# This expects the folder to have state.json, dataset_info.json, and *.arrow files
raw_dataset = load_from_disk(dataset_path)

print(f"Dataset loaded. Structure: {raw_dataset}")

# 2. LOAD MODEL & TOKENIZER
# ---------------------------------------------------------
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. TOKENIZE THE DATASET
# ---------------------------------------------------------
# We need to identify the column name containing the text (usually "text" or "content")
column_names = raw_dataset.column_names
text_column = "text" if "text" in column_names else column_names[0]
print(f"Using column '{text_column}' as input text.")

def tokenize_function(examples):
    # Add EOS token to the end of every text so the model knows where sentences end
    # We assume the dataset has a column named 'text' (or whatever text_column found)
    texts = [t + tokenizer.eos_token for t in examples[text_column]]
    
    output = tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    # GPT-2 labels are the same as input_ids
    output["labels"] = output["input_ids"].copy()
    return output

# Apply tokenization to the whole dataset efficiently
# batched=True processes multiple rows at once for speed
tokenized_dataset = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=column_names # Remove raw text columns to leave only tensors
)

# If the loaded dataset is a DatasetDict (has train/test keys), select 'train'
if hasattr(tokenized_dataset, "keys") and "train" in tokenized_dataset.keys():
    tokenized_dataset = tokenized_dataset["train"]

# Ensure format is torch for the trainer
tokenized_dataset.set_format("torch")

# 4. SETUP TRAINING
# ---------------------------------------------------------
model = GPT2LMHeadModel.from_pretrained(model_name)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./pretrained_model_state",
    overwrite_output_dir=True,
    learning_rate=2e-5, 
    num_train_epochs=3,        
    weight_decay=0.01,
    warmup_steps=100,
    per_device_train_batch_size=2, # Keep low if VRAM is tight
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    use_cpu=False if torch.cuda.is_available() else True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 5. TRAIN AND SAVE
# ---------------------------------------------------------
print("--- Starting Training ---")
trainer.train()
print("--- Training Finished ---")

output_save_path = "./model_shard_trained"
model.save_pretrained(output_save_path)
tokenizer.save_pretrained(output_save_path)
print(f"Model saved to {output_save_path}")

# 6. VERIFICATION
# ---------------------------------------------------------
print("\n--- Testing ---")
model.eval()
input_text = "CosmosTech şirketi"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    inputs.input_ids, 
    max_length=1024, 
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")