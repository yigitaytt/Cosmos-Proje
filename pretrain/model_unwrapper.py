import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config
from pathlib import Path


def unwrap_compiled_checkpoint(input_path, output_path):
    """
    Unwrap a torch.compile() checkpoint and save it properly.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"Loading checkpoint from: {input_path}")
    
    # Load config and tokenizer
    config = GPT2Config.from_pretrained(input_path)
    tokenizer = AutoTokenizer.from_pretrained(input_path)
    
    # Load the wrapped state dict
    checkpoint_file = input_path / "pytorch_model.bin"
    if not checkpoint_file.exists():
        checkpoint_file = input_path / "model.safetensors"
        if checkpoint_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_file)
        else:
            raise FileNotFoundError("No model file found!")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")
    
    print(f"Loaded {len(state_dict)} keys from checkpoint")
    
    # Remove _orig_mod prefix from all keys
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        unwrapped_state_dict[new_key] = value
    
    # Handle tied weights: lm_head.weight should be the same as wte.weight
    if 'lm_head.weight' not in unwrapped_state_dict and 'transformer.wte.weight' in unwrapped_state_dict:
        print("Adding lm_head.weight (tied with transformer.wte.weight)")
        unwrapped_state_dict['lm_head.weight'] = unwrapped_state_dict['transformer.wte.weight']
    
    print(f"Unwrapped to {len(unwrapped_state_dict)} keys")
    print("Sample unwrapped keys:")
    for i, key in enumerate(list(unwrapped_state_dict.keys())[:5]):
        print(f"  {key}")
    
    # Create fresh model and load unwrapped weights
    print("\nCreating model and loading weights...")
    model = GPT2LMHeadModel(config)
    
    # Load the unwrapped weights
    missing_keys, unexpected_keys = model.load_state_dict(unwrapped_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    else:
        print("✓ No missing keys!")
        
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys}")
    else:
        print("✓ No unexpected keys!")
    
    # Save the clean model
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving unwrapped model to: {output_path}")
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("✓ Done! Model unwrapped and saved successfully.")
    return model, tokenizer


def test_unwrapped_model(model_path, prompt="Bir matematik problemi:"):
    """
    Load and test the unwrapped model.
    """
    print("\n" + "="*60)
    print("Testing the unwrapped model...")
    print("="*60)
    
    # Load the unwrapped model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if it loaded properly
    print(f"\nModel loaded successfully!")
    print(f"lm_head weight shape: {model.lm_head.weight.shape}")
    print(f"wte weight shape: {model.transformer.wte.weight.shape}")
    print(f"Weights are tied: {model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()}")
    
    # Test generation
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    model.to(device)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*60)
    print("Generated text:")
    print("="*60)
    print(result)
    print("="*60)
    
    return model, tokenizer


def main():
    """
    Main function to unwrap checkpoint and test model.
    """
    # Configuration
    input_checkpoint = "./agressive_token/final_model"
    output_checkpoint = "./agressive-token-unwrapped"
    
    # Unwrap the checkpoint
    model, tokenizer = unwrap_compiled_checkpoint(input_checkpoint, output_checkpoint)
    
    # Test the unwrapped model
    print("\n" + "="*60)
    print("Testing immediately after unwrapping...")
    print("="*60)
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    prompt = "Bir matematik problemi:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{result}")
    print("\n" + "="*60)
    
    # Test by reloading from disk
    test_unwrapped_model(output_checkpoint, prompt="Matematik:")


if __name__ == "__main__":
    main()