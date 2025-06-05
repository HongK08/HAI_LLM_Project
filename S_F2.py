import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from huggingface_hub import login

login(token='')

# 경로 설정
BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DATA_PATH = "/home/user/바탕화면/Medical_CONF/formatted_medical_dataset.jsonl"
OUTPUT_DIR = "./exaone13b-kormed-lora"
REPO_ID = "HongKi08/HAI_Project"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompt(example):
    return {
        "prompt": example["instruction"],
        "output": example["output"]
    }

formatted_data = dataset.map(format_prompt)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# LoRA For EXAONE
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # EXAONE 계열에서 일반적으로 사용
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 토큰화 
def tokenize(example):
    return tokenizer(
        example["prompt"],
        text_target=example["output"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )

tokenized_dataset = formatted_data.map(tokenize, batched=False)

# Train Setup
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none"
)

# Trainer exec
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# safetensor save
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# Push -> Hugging Face
model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)
