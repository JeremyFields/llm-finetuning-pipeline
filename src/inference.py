import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Configuration
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"
ADAPTER_DIR = "./lora_adapter"

print("Loading base model and tokenizer...")

# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)x
tokenizer.pad_token = tokenizer.eos_token

# 3. Load the Base Model in 4-bit (Just like training, save VRAM for my local RTX)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Attach custom LoRA Adapter
print(f"Attaching LoRA adapter from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("\n" + "="*50)
print("🧠 Model is ready! Type 'exit' to quit.")
print("="*50 + "\n")

# 5. Interactive Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
        
    formatted_prompt = f"<s>[INST] {user_input} [/INST] "
    
    # Convert text to tokens
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens=150, # Limit response length
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Convert tokens back to text
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\nAI: {response.strip()}\n")