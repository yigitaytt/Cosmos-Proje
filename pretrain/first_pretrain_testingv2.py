import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # 512'den 256'ya indirmek hızı 2 kat artırır
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

def main():
    dataset_path = r"C:\Users\pc\OneDrive - Yildiz Technical University\Desktop\Cosmos\Cosmos-Proje\pretrain\data\shard_0"

    print(f"Loading dataset from: {dataset_path}")

    raw_dataset = load_from_disk(dataset_path)
    
    # GPU Kontrolü
    print(f"Aktif GPU Sayısı: {torch.cuda.device_count()}")

    # Model Hazırlığı
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    column_names = raw_dataset.column_names
    
    # 3. Apply Map with Multiprocessing
    tokenized_datasets = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=column_names,
        num_proc=4
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Agresif ve Hızlı Eğitim Ayarları
    training_args = TrainingArguments(
        output_dir="./turkish-gpt2-math-model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,      
        gradient_accumulation_steps=4,      
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,                          
        logging_steps=10,                   
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        ddp_find_unused_parameters=False    
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )

    print("Eğitim başlıyor...")
    trainer.train()

    trainer.save_model("./output_model")
    tokenizer.save_pretrained("./output_model")

if __name__ == "__main__":
    main()