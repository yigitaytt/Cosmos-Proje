# !pip install git+https://github.com/huggingface/trl.git # trl için genelde gerekli oluyor. Çalıştırılan ortama göre kontrol edilip indirilmeli
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

# ==========================================
# 1. AYARLAR VE MODEL YÜKLEME
# ==========================================
# NOT: Bu kod çalıştırılmadan önce aşağıdaki yollar güncellenmelidir.

# TODO: SFT eğitimi bitmiş modelin yolu 
model_name = "path/to/your/sft-model" 

# TODO: Eğitim verisi 
dataset_path = "path/to/your/dataset.jsonl" 

output_dir = "grpo_results"

# GPU durumuna göre dtype ayarı
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Model ve Tokenizer yükleniyor...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# GRPO üretim (generation) yaptığı için padding SOL tarafta olmalı
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

print(f"Model yüklendi: {model_name}")

# ==========================================
# 2. VERİ SETİ HAZIRLIĞI
# ==========================================
print(f"Veri seti yükleniyor: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Train/Test ayırma (%5 test yeterli)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# --- Prompt Formatlama ---
# GRPO'ya giren veriyi SFT modelinin alıştığı formata (### Question...) çeviriyoruz.
def format_prompt(example):
    return {
        "prompt": f"### Question:\n{example['prompt']}\n\n### Answer:\n"
    }

train_dataset = train_dataset.map(format_prompt)
eval_dataset = eval_dataset.map(format_prompt)

print(f"Eğitim Örneği: {len(train_dataset)}")
print(f"Test Örneği:   {len(eval_dataset)}")

# ==========================================
# 3. ÖDÜL FONKSİYONLARI (SFT SONRASI DURUMA GÖRE FORMAT DEĞİŞİKLİĞİ YAPILABİLİR)
# ==========================================

def extract_answer_value(text):
    """
    Modelin cevabından sayıyı çeker.
    Model '### Answer:' sonrası konuşacak. Biz son üretilen sayıyı veya
    varsa 'Cevap: X' formatını arayacağız.
    """
    # 1. Sadece modelin ürettiği kısmı al (Prompt'u at)
    if "### Answer:" in text:
        generated_part = text.split("### Answer:")[-1].strip()
    else:
        generated_part = text
    
    # 2. Cevap satırını ara 
    return generated_part

def parse_number(text):
    """String'i (örn: '1/2' veya '0.5') float'a çevirir."""
    if not text: return None
    try:
        # Metnin içindeki tüm sayıları bul
        matches = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
        if matches:
            last_num = matches[-1] # En sondaki sayıyı cevap kabul et
            if "/" in last_num:
                n, d = last_num.split("/")
                return float(n) / float(d)
            return float(last_num)
    except:
        pass
    return None

# Ödül 1: Format Koruma Ödülü
def format_reward_func(completions, **kwargs):
    """
    Modelin boş cevap verip vermediğini kontrol eder.
    SFT formatını zaten prompt ile zorladığımız için tag aramaya gerek yok.
    """
    rewards = []
    for text in completions:
        # Metin boş değilse ve en azından bir sayı veya işlem içeriyorsa ödül ver
        if len(text.strip()) > 0 and any(char.isdigit() for char in text):
            rewards.append(0.5) 
        else:
            rewards.append(0.0)
    return rewards

# Ödül 2: Reasoning (Uzun Düşünme) Ödülü
def reasoning_reward_func(completions, **kwargs):
    """Cevabı hemen yapıştırmak yerine biraz açıklama yazarsa ödül ver."""
    rewards = []
    for text in completions:
        if "### Answer:" in text:
            gen_part = text.split("### Answer:")[-1].strip()
            # Eğer cevap 50 karakterden uzunsa ödül ver
            if len(gen_part) > 50: 
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# Ödül 3: Doğruluk Kontrolü
def correctness_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        # Modelin ürettiği metni al
        gen_text = extract_answer_value(completion)
        
        # Sayıya çevir 
        gen_val = parse_number(gen_text)
        
        # Gerçek cevabı sayıya çevir
        truth_val = parse_number(str(ground_truth))
        
        if gen_val is not None and truth_val is not None:
            # Toleranslı karşılaştırma
            if abs(gen_val - truth_val) < 1e-4:
                rewards.append(2.0) # Doğru Cevap
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
            
    return rewards

# ==========================================
# 4. LORA & GRPO AYARLARI
# ==========================================

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,  # Overfit engellemek için %5 rastgele kapatma
    target_modules=["c_attn", "c_proj", "c_fc"],
    bias="none",
)

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,  # Sft'den daha düşük lr olmalı       
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1, # Verinin ilk %10'luk kısmı warmup
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    
    # GRPO Ayarları
    num_generations=8,  # Soru başına kaç cevap üretilecek. Test için VRAM yetmez ise azaltılabilir.       
    max_prompt_length=512,
    max_completion_length=512, 
    beta=0.04,                 
    
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True,
)

# ==========================================
# 5. TRAINER VE EĞİTİM
# ==========================================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward_func, reasoning_reward_func, correctness_reward_func], # Ödül fonksiyonları 
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config
)

print("\nGRPO Eğitimi Başlıyor ")
print("-" * 50)

trainer.train()

# ==========================================
# 6. KAYDETME
# ==========================================
print(f"\nModel kaydediliyor: {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Eğitim Bitti!")