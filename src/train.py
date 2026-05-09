import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 1. Connect to your Local MLflow Container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("12gb-gpu-lora-finetuning")

# 2. Load Model in 4-bit (Crucial for the RTX 3060)
# We'll use a small, fast model for the proof-of-concept
model_id = "Qwen/Qwen2.5-1.5B" 

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# bitsandbytes handles the 4-bit loading automatically under the hood
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
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
dataset = load_dataset("json", data_files="../data/sample_dataset.jsonl", split="train")

# 5. Training Arguments (Optimized for 12GB)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1, # Keep it at 1 so we don't crash the GPU
    gradient_accumulation_steps=4, # Simulates a larger batch size logically
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=2e-4,
    fp16=True, # RTX 3060 supports mixed precision
    max_steps=10, # Keep it short for the test run
    report_to="mlflow" # Automatically sends metrics to localhost:5000
)

# 6. Execute the Pipeline
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

print("Starting training. Watch MLflow at localhost:5000!")
trainer.train()

# 7. Save the resulting weights
trainer.model.save_pretrained("lora_adapter")
print("Pipeline complete. Adapter saved.")