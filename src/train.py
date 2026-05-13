import mlflow
import torch
import os
from textwrap import dedent
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

# Longer training and more MLflow stats for a portfolio-friendly run.
TRAIN_EPOCHS = 5
EVAL_SPLIT = 0.05
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10
WARMUP_STEPS = 30

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

# 4. Load the data
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")
splits = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]

sft_config = SFTConfig(
    output_dir="./results",
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4, 
    learning_rate=2e-4,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    num_train_epochs=TRAIN_EPOCHS,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    fp16=False,
    optim="paged_adamw_32bit",
    report_to="mlflow",
    gradient_checkpointing=True,
)

# 6. Initialize Trainer
with mlflow.start_run(run_name=f"lora-qwen2.5-{TRAIN_EPOCHS}epochs"):
    mlflow.log_params(
        {
            "model_id": model_id,
            "train_epochs": TRAIN_EPOCHS,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "learning_rate": 2e-4,
            "warmup_steps": WARMUP_STEPS,
            "max_length": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
        }
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    print(f"CUDA status: {torch.cuda.is_available()}")
    print(f"Model is on: {model.device}")

    print("Starting training. Watch MLflow at localhost:5000!")
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    mlflow.log_metrics(
        {
            "train_loss": float(train_result.training_loss),
            "eval_loss": float(eval_result["eval_loss"]),
        }
    )

    # 7. Save the resulting weights
    trainer.model.save_pretrained("lora_adapter")
    tokenizer.save_pretrained("lora_adapter")
    mlflow.log_artifacts("lora_adapter", artifact_path="lora_adapter")

    sample_prompt = dedent(
        """
        Human: What is MLOps?
        Assistant:
        """
    ).strip()
    sample_inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
    sample_outputs = trainer.model.generate(
        **sample_inputs,
        max_new_tokens=120,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    sample_response = tokenizer.decode(
        sample_outputs[0][sample_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    mlflow.log_text(
        f"Prompt:\n{sample_prompt}\n\nResponse:\n{sample_response}",
        artifact_file="sample_generation.txt",
    )

    print("Pipeline complete. Adapter saved and logged to MLflow.")