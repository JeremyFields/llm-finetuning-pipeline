import  mlflow
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# Set Paths for 
current_dir = Path(__file__).parent 
data_path = current_dir.parent / "data" / "sample_dataset.jsonl"

# 1. Connect to your Local MLflow Container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("12gb-gpu-lora-finetuning")

# 2. Load Model in 4-bit (Crucial for the RTX 3060)
# We'll use a small, fast model for the proof-of-concept
model_id = "Qwen/Qwen2.5-1.5B" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Load Model with the config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# 3. Setup LoRA (Only train ~1% of the model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 4. Load the DVC-Tracked Data
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

sft_config = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, 
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=100,
    bf16=True,
    fp16=False,
    optim="paged_adamw_32bit",
    report_to="mlflow",
    save_strategy="no"
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

print(f"CUDA status: {torch.cuda.is_available()}")
print(f"Model is on: {model.device}")

print("Starting training. Watch MLflow at localhost:5000!")
trainer.train()

# 7. Save the resulting weights
trainer.model.save_pretrained("lora_adapter")
print("Pipeline complete. Adapter saved.")