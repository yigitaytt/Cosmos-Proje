from transformers import AutoTokenizer, GPT2LMHeadModel
model_name = "ytu-ce-cosmos/turkish-gpt2-medium"

# This downloads and saves the model to a specific folder
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./ytu-ce-cosmos/turkish-gpt2-medium")
tokenizer.save_pretrained("./ytu-ce-cosmos/turkish-gpt2-medium")