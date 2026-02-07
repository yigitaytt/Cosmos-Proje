import os
import sys
import logging
import gc
import re
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

# ============================================================================
# 1. INDUSTRIAL LOGGING & PATH CONFIGURATION (UHEM STANDARDS)
# ============================================================================
MODEL_NAME = os.getenv('MODEL_NAME', 'ytu-ce-cosmos/turkish-gpt2-medium')
DATASET_FILE = os.getenv('DATASET_NAME', 'uhem_final_ready.json') 
WORKSPACE = Path(os.getenv('TRAINING_WORKSPACE', '/home/user/uhem_work'))
JOB_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

OUTPUT_DIR = WORKSPACE / 'jobs' / JOB_ID / 'output'
CHECKPOINT_DIR = WORKSPACE / 'jobs' / JOB_ID / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ============================================================================
# 2. INTEGRITY CHECK (LoRA Verification)
# ============================================================================
def verify_lora_integrity(model):
    trainable_params = 0
    all_param = 0
    lora_layers_found = False
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora_" in name:
            lora_layers_found = True

    ratio = 100 * trainable_params / all_param
    
    logging.info("="*50)
    logging.info(f"{'LoRA INTEGRITY CHECK':^50}")
    logging.info("="*50)
    logging.info(f"Trainable Params: {trainable_params:,}")
    logging.info(f"All Params      : {all_param:,}")
    logging.info(f"Trainable Ratio : %{ratio:.4f}")
    
    if not lora_layers_found or trainable_params == 0:
        raise RuntimeError("❌ HATA: LoRA katmanları aktif değil!")
    logging.info("✅ LoRA doğrulaması başarılı.")
    logging.info("="*50)

# ============================================================================
# 3. OFFSET-BASED CUSTOM COLLATOR (Milimetrik Maskeleme)
# ============================================================================
class StrictMathCollator:
    def __init__(self, tokenizer, response_template="#### Cevap:"):
        self.tokenizer = tokenizer
        self.response_template = response_template

    def __call__(self, examples):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for item in examples:
            inst = item["instruction"]
            out = item["output"]

            # --- FORMAT ---
            if self.response_template in inst:
                text = f"{inst}{out}{self.tokenizer.eos_token}"
            else:
                text = (
                    f"#### Soru: {inst}\n"
                    f"{self.response_template} {out}{self.tokenizer.eos_token}"
                )

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=1024,
                padding=False,
                return_offsets_mapping=True
            )

            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            labels = [-100] * len(input_ids)

            # --- CEVAP BAŞLANGICI ---
            answer_char_start = text.index(self.response_template) + len(self.response_template)

            # boşluğu da atla
            while answer_char_start < len(text) and text[answer_char_start] == " ":
                answer_char_start += 1

            for i, (start, end) in enumerate(offsets):
                if start >= answer_char_start:
                    labels[i] = input_ids[i]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_labels.append(torch.tensor(labels))
            batch_attention_mask.append(torch.tensor(enc["attention_mask"]))

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                batch_labels,
                batch_first=True,
                padding_value=-100
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                batch_attention_mask,
                batch_first=True,
                padding_value=0
            ),
        }
    
# ============================================================================
# 4. MAIN PIPELINE (UHEM Training Arguments)
# ============================================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logging.info(f"Veri yükleniyor: {DATASET_FILE}")
    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    split_ds = dataset.train_test_split(test_size=0.01, seed=42)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj", "c_fc"]
    )
    model = get_peft_model(model, peft_config)
    verify_lora_integrity(model)

    collator = StrictMathCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["test"],
        data_collator=collator,
    )

    # Sanity Check
    logging.info("Sanity Check Yapılıyor...")
    sample_batch = collator([split_ds["train"][0]])
    trained_tokens = [sample_batch["input_ids"][0][i].item() for i in range(len(sample_batch["labels"][0])) if sample_batch["labels"][0][i] != -100]
    logging.info(f"ÖĞRENECEĞİ KISIM:\n{tokenizer.decode(trained_tokens)}")

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final_model_lora"))

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()