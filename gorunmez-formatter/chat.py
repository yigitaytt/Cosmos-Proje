from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./final_model" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda") # cuda CPU yerine GPU kullandirtmak icin 

# 2. THE BUTLER FUNCTION 
def ask_the_bot(user_question):

    prompt = f"### Question:\n{user_question}\n\n### Answer:\n"   # otomatik girilen promptun basina ve sonuna modelin ezberledigi 
    # 3. Send to Model                                              ve bekledigi formati basiyoruz
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # text IDlere donusturuldu
    # 4. Generate Response
    outputs = model.generate(
        **inputs,             # ID erisimi
        max_new_tokens=100,   # cevap token limiti degisebilir
        do_sample=True,       # bu false ise otomatik en cok beklenen kelimeyi yerlestiriyo ama true ise en cok beklenenler arasindan biri 
        temperature=0.7,      # do_sample true iken ne derece kesin cevap verme ayari 0.1 olursa robotik, 1.0 olursa cok kolpa 
        pad_token_id=tokenizer.eos_token_id # Fix for GPT-2 end of sentences basmaca
    )
    
    # 5. Decode the raw numbers back to text
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 6. Clean up: The model repeats the question. We usually want just the answer.
    # We split by "### Answer:" and take the second part
    answer_only = full_text.split("### Answer:\n")[-1].strip()
    
    return answer_only

# --- NOW USE IT ---
while True:                                         # kullanicidan input gelene kadar bekleme 
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:       # bu anahtar kelimelerden biri kullanilirsa otomatik sonlandirma
        break
        
    response = ask_the_bot(user_input)
    print(f"Bot: {response}")
    print("-" * 30)