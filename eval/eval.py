import os
import torch
import math
import logging
import re
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

# ============================================================================
# 1. AYARLAR VE YOLLAR
# ============================================================================
BASE_MODEL_ID = "ytu-ce-cosmos/turkish-gpt2-medium"
NEW_MODEL_PATH = "/kaggle/input/agressive-math-token-model-checkpoint-6000/pytorch/default/1/checkpoint-6000-unwrapped"
DATASET_FILE = "/kaggle/input/deneme/diverse_plain_sft.json"

BATCH_SIZE = 4
MAX_SEQ_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# 2. PREPROCESSOR
# ============================================================================
def global_math_preprocessor(text: str) -> str:
    if not isinstance(text, str): return ""
    bad_words = "milyon|ay|Ay|tane|birim|adet"
    pattern = rf'(\d+)\s*({bad_words})(?=\s*[+\-*/=]|\s*\d)'
    text = re.sub(pattern, r'\1x', text)
    text = re.sub(r'([+\-*/=()])', r' \1 ', text)
    return " " + " ".join(text.split())

# ============================================================================
# 3. EVALUATION ENGINE (LOSS & PPL)
# ============================================================================
def evaluate_specific_model(path_or_id, dataset):
    logging.info(f"\n--- LOSS/PPL HESAPLANIYOR: {path_or_id} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(path_or_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path_or_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    response_template = "### Cevap:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    
    total_loss = 0.0
    total_active_tokens = 0
    correct_top1 = 0
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum')

    with torch.inference_mode():
        for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
            batch = dataset[i : i + BATCH_SIZE]
            
            prompts = []
            for q, a in zip(batch['question'], batch['answer']):
                q_p = global_math_preprocessor(q)
                a_p = global_math_preprocessor(a)
                text = f"### Soru:\n{q_p}\n\n### Cevap:\n{a_p}{tokenizer.eos_token}"
                prompts.append(text)
            
            encodings = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
            input_ids = encodings.input_ids.to(DEVICE)
            attention_mask = encodings.attention_mask.to(DEVICE)
            labels = input_ids.clone()

            for j in range(labels.size(0)):
                input_list = input_ids[j].tolist()
                idx = -1
                # Basit liste araması (daha hızlı olabilir)
                for k in range(len(input_list) - len(response_template_ids)):
                    if input_list[k : k + len(response_template_ids)] == response_template_ids:
                        idx = k + len(response_template_ids)
                        break
                labels[j, :idx] = -100 if idx != -1 else -100

            outputs = model(input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            active_mask = (shift_labels != -100)
            num_active = active_mask.sum().item()
            
            if num_active > 0:
                total_loss += loss.item()
                total_active_tokens += num_active
                preds = shift_logits.view(-1, shift_logits.size(-1))[active_mask.view(-1)].argmax(dim=-1)
                correct_top1 += (preds == shift_labels.view(-1)[active_mask.view(-1)]).sum().item()

    results = {
        "Loss": total_loss / total_active_tokens,
        "PPL": math.exp(min(total_loss / total_active_tokens, 100)),
        "Acc": (correct_top1 / total_active_tokens) * 100
    }
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

# ============================================================================
# 4. GENERATION & EXACT MATCH CHECK (DÜZELTİLEN KISIM)
# ============================================================================
def verify_exact_match(model_path, dataset, device):
    """
    Modeli yükler, generate eder ve cevabı regex ile karşılaştırır.
    """
    print(f"\n--- GENERATION TEST YÜKLENİYOR: {model_path} ---")
    
    # Modeli ve Tokenizer'ı burada yüklüyoruz (Main scope'tan bağımsız)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    correct = 0
    # Test edilecek örnek sayısı (Çok uzun sürmemesi için 50 ile sınırladım)
    num_samples = min(50, len(dataset))
    
    print(f"🧐 GERÇEK ÇÖZÜM KONTROLÜ BAŞLADI ({num_samples} örnek)...")
    
    for i in tqdm(range(num_samples)):
        q = dataset[i]['question']
        # Dataset'teki orijinal cevap (raw string olabilir veya içinde #### olabilir)
        a_true_raw = str(dataset[i]['answer'])
        
        # Preprocessing (Model spaced digit bekliyor)
        q_p = global_math_preprocessor(q)
        
        prompt = f"### Soru:\n{q_p}\n\n### Cevap:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=64, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False  # Greedy decoding (en kesin cevap için)
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # --- CEVAP AYIKLAMA (REGEX) ---
        # Senin modelin "1 2 . 5" gibi boşluklu çıktı veriyor.
        # Regex: #### sonrası rakam, boşluk, nokta veya virgül içeren grubu yakalar.
        def extract_answer(text):
            # #### işaretinden sonrasını al
            if "####" not in text: return None
            after_hash = text.split("####")[-1].strip()
            
            # Sadece sayısal değerleri temizleyip al (boşlukları sil)
            # Örn: "1 2 . 5" -> "12.5"
            clean_num = after_hash.replace(" ", "").replace(",", ".")
            # Sadece sayısal karakterleri ve noktayı bırak
            clean_num = re.sub(r"[^\d.]", "", clean_num)
            return clean_num

        model_ans = extract_answer(generated_text)
        true_ans = extract_answer(a_true_raw) 
        
        # Eğer true_ans #### ile bulunamazsa (dataset yapısına göre), raw text temizlenir
        if true_ans is None:
             # Basitçe raw text'ten sayı çekmeyi dene (Fallback)
             clean_raw = a_true_raw.replace(" ", "").replace(",", ".")
             digits = re.findall(r"[-+]?\d*\.?\d+", clean_raw)
             true_ans = digits[-1] if digits else "YOK"

        if model_ans == true_ans and true_ans != "YOK":
            correct += 1
            status = "✅"
        else:
            status = "❌"
            
        if i < 3: # İlk 3 örneği detaylı bas
            print(f"\nSoru: {q[:50]}...")
            print(f"Model Çıktısı (Raw): {generated_text.split('### Cevap:')[-1][:50]}...")
            print(f"Model Sayı: {model_ans} | Gerçek Sayı: {true_ans} -> {status}")

    acc = (correct / num_samples) * 100
    print(f"\n🎯 TAM EŞLEŞME SKORU (Exact Match): %{acc:.2f}")
    
    # Temizlik
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# 5. ÇALIŞTIRMA VE KARŞILAŞTIRMA
# ============================================================================
def main():
    print(f"📊 Veri yükleniyor: {DATASET_FILE}")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    # --- MODEL 1: BASE MODEL (Sadece Loss/PPL bakılır) ---
    base_res = evaluate_specific_model(BASE_MODEL_ID, dataset)

    # --- MODEL 2: YENİ MODEL (Loss/PPL + Generation Test) ---
    new_res = evaluate_specific_model(NEW_MODEL_PATH, dataset)
    
    # Parametreleri düzelttik: Artık path ve dataset alıyor
    verify_exact_match(NEW_MODEL_PATH, dataset, DEVICE)
    
    # --- FINAL RAPOR ---
    print("\n" + "="*60)
    print(f"{'METRİK':<20} | {'BASE MODEL':<15} | {'YENİ MODEL':<15}")
    print("-"*60)
    print(f"{'Avg Loss':<20} | {base_res['Loss']:>15.4f} | {new_res['Loss']:>15.4f}")
    print(f"{'Perplexity':<20} | {base_res['PPL']:>15.4f} | {new_res['PPL']:>15.4f}")
    print(f"{'Top-1 Accuracy':<20} | {base_res['Acc']:>15.2f} | {new_res['Acc']:>15.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
