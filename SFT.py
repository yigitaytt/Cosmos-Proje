from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

# 1. Load your Model and Tokenizer
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-medium")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)    # model yuklendi 
model.resize_token_embeddings(len(tokenizer))  # modelin embedding matrixinin row sayisiyla tokenizerin row sayisi arasinda uyusmazlik olmasin 
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Load your Dataset
# It must have 'question' and 'answer' columns
dataset = load_dataset("json", data_files="Final_Turkish_Math_Mix_130k.jsonl", split="train")


dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

print(f"Training Samples: {len(train_dataset)}")
print(f"Validation Samples: {len(eval_dataset)}")



# 3. DEFINE THE "RECIPE" (Updated for Question + Answer)
                                            # 23-56 line arasinda sadece formatlama isleri yapiliyo
question_col = ""                           # normalde data setten Question ve Answer baslikli column bekliyodu
answer_col = ""                             # artik possible_name olarak girilen tum inputlari bekliyo oraya ekleme yapmak cok daha kolay
# Look for common names for the "Question" part
for possible_name in ["question", "query", "instruction", "input"]:
    if possible_name in train_dataset.column_names:
        question_col = possible_name
        break
# Look for common names for the "Answer" part
for possible_name in ["answer", "response", "output", "target"]:
    if possible_name in train_dataset.column_names:
        answer_col = possible_name
        break
# Verification
print(f"Detected Question Column: '{question_col}'")
print(f"Detected Answer Column:   '{answer_col}'")
# Stop if we couldn't find them
if not question_col or not answer_col:
    raise ValueError("Could not automatically find question/answer columns!")
# 4. DEFINE THE GENERIC FUNCTION
def formatting_prompts_func(example):
    output_texts = []
    for q, a in zip(example[question_col], example[answer_col]):
        formatted_text = f"### Question:\n{q}\n\n### Answer:\n{a}" + tokenizer.eos_token 
        output_texts.append(formatted_text)
        
    return output_texts

# 5. Configure the Training
sft_config = SFTConfig(
    output_dir="./results",
    packing=True,           # bunu true yaparak kisa cumleleri gereksiz paddingle doldurmak yerine uc uca ekleyip modeli gereksiz yormuyoruz
    max_seq_length=1024,       # packkingin optimum boyutu modele ve datasete bagli degisiyo GPT-2 modelleri icin 1024 en uygunu gibi gozukuyo 
    per_device_train_batch_size=2,  # dusuk batch kullanip daha az vram kullanarak ve daha az noise ureterek hesaplama yapiyo
    gradient_accumulation_steps=8,  # her 2 adimda updatelemek yerine 8 defa 2li agirlik alip ortalamalarina gore kendini updateliyo AdaGrad
    learning_rate=2e-5,
    num_train_epochs=3,     # burdaki 3luyle duruma gore oynayabiliriz
    logging_steps=10, 
    bf16=True,      # normalde 32bitlik fp kullanilirken bunu 16'ya dusurup yaklasik %50 memory ve hizdan kazaniyo
                    # bunu her sey icin yapmiyo onemli kisimlari hala 32likte yapiyo 
                    # eger your hardware doesn t support bf16 uyarisi alinirsa bf16 yerin fp16 yazilacak
    eval_strategy="steps",
    eval_steps=100,                 # adim adim sonuc izlemek icin
    save_steps=100,                 
    save_total_limit=2,             # geriden 2 tane chekpoint tutuyo
    load_best_model_at_end=True,    # eger olurda 3. epoch sirasinda overfit yasanirsa gecmisteki en iyi halden devam
    metric_for_best_model="eval_loss",
    weight_decay=0.01,              # overfit engellemek icin weight kirpma             
)                                    


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=64,             #  ogrenmenin buyuklugunu ayarliyo. uhem var performans sorunu az olacagindan 64e yukselttim
    lora_alpha=128,    
    lora_dropout=0.1,  # overfit engellemek icin rastgele %10luk veriyi atiyo
    target_modules=["c_attn", "c_proj", "c_fc"] 
)



# 6. Create the Trainer
trainer = SFTTrainer(    # ayarlarini yaptigimiz parametreleri yerine koyup calistirmaya geciyoz
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,    
    args=sft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer, 
    peft_config=peft_config,
)

# 7. Start Training
trainer.train()


# 8. Save the Final Model                           # bu kismin amaci training sonucunun chat.py ye yollanmasi 
trainer.save_model("./final_model")                 # eger chat.py iptal olursa silinebilir 
tokenizer.save_pretrained("./final_model")
print("Training finished. Model saved to ./final_model")