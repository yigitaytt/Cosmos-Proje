import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn import CrossEntropyLoss

# ==========================================
# 1. AYARLAR
# ==========================================
# TODO: SFT sonrası modelinin yolu
model_name = "path/to/your/sft-model"

# TODO: Offline Dataset
dataset_path = "grpo_offline.jsonl" 

output_dir = "grpo_offline_results"

max_seq_length = 1024
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ==========================================
# 2. DATASET (VERİYİ OKUMA)
# ==========================================
class OfflineGRPODataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        print(f" Veri seti işleniyor: {data_path}")
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # Datasetindeki "responses" listesi
                if "responses" not in item: continue 
                
                prompt = item["prompt"]
                responses = item["responses"] 
                
                # --- GRUP İÇİ AVANTAJ HESABI ---
                scores = [r["score"] for r in responses]
                mean_score = np.mean(scores)
                std_score = np.std(scores) + 1e-8 # Sayı/0 olmaması için ufak bir sayı eklendi
                
                for resp, score in zip(responses, scores):
                    # Avantaj: (Puan - Ort) / Std
                    advantage = (score - mean_score) / std_score
                    
                    self.data.append({
                        "prompt": prompt,
                        "completion": resp['text'], # Datasetten gelen cevap metni
                        "advantage": float(advantage)
                    })
                    
        print(f" Toplam Örnek: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 3. COLLATOR (FORMATLAMA VE MASKELEME)
# ==========================================
class GRPODataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Kodun içine eklediğimiz Şablon
        self.response_template = "### Answer:\n"
        
        # Bu ayracın token ID'lerini buluyoruz (Maskeleme için referans noktası)
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, 
            add_special_tokens=False
        )

    def __call__(self, batch):
        prompts = [x['prompt'] for x in batch]
        completions = [x['completion'] for x in batch]
        advantages = [x['advantage'] for x in batch]

        full_texts = [f"### Question:\n{p}\n\n{self.response_template}{c}" for p, c in zip(prompts, completions)]
        
        # Tokenize et
        tokenized = self.tokenizer(
            full_texts,
            padding=True,          
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True 
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone() 

        # --- MASKELEME ---
        # "### Answer:\n" ifadesini bulup ÖNCESİNİ kapatacağız.
        for i in range(len(input_ids)):
            # Şablonu (Response Template) metin içinde ara
            response_start_idx = -1
            len_template = len(self.response_token_ids)
            
            # input_ids içinde şablon dizisini arıyoruz
            for j in range(len(input_ids[i]) - len_template):
                if input_ids[i][j : j + len_template].tolist() == self.response_token_ids:
                    response_start_idx = j + len_template # Şablonun bittiği yer (Cevapların başladığı yer)
                    break
            
            # Bulduysan maskele
            if response_start_idx != -1:
                labels[i, :response_start_idx] = -100 # Başlangıçtan -> Cevaba kadar KAPAT
            else:
                # Şablon yoksa (çok nadir) hepsini kapat
                labels[i, :] = -100 

            # Paddingleri de kapat
            labels[i][attention_mask[i] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantage": torch.tensor(advantages, dtype=torch.float32)
        }

# ==========================================
# 4. TRAINER 
# ==========================================
class OfflineGRPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        advantages = inputs.get("advantage")
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.get("logits")
        
        # --- LOSS HESABI ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(reduction='none') 
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_loss = token_loss.view(shift_labels.size())
        
        valid_mask = (shift_labels != -100).float()
        sentence_loss = (token_loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
        
        # Cihaz uyumu
        advantages = advantages.to(sentence_loss.device)

        # Advantage değerlerini -5 ile +5 arasına sıkıştırıyoruz.
        # Bu, kötü cevapların loss'u "eksi sonsuza" çekip eğitimi patlatmasını engeller.

        clamped_advantages = torch.clamp(advantages, min=-5.0, max=5.0)

        # GRPO: Loss = Hata * Avantaj
        weighted_loss = sentence_loss * clamped_advantages
        final_loss = weighted_loss.mean()
        
        return (final_loss, outputs) if return_outputs else final_loss

# ==========================================
# 5. BAŞLAT
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto"
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "c_fc"], 
    bias="none",
)
model = get_peft_model(model, peft_config)

train_dataset = OfflineGRPODataset(dataset_path)
collator = GRPODataCollator(tokenizer, max_length=max_seq_length)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,          
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=4,
    num_train_epochs=1,          
    fp16=True if torch.cuda.is_available() else False,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    optim="adamw_torch",
    remove_unused_columns=False, 
    report_to="none"             
)

trainer = OfflineGRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator
)

print("\nOffline GRPO Başlıyor...")
trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Bitti!")