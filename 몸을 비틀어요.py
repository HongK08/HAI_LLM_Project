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

# Hugging Face 로그인
login(token='')

# 경로 설정
BASE_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DATA_PATH = "/home/user/바탕화면/Medical_CONF/formatted_medical_dataset.jsonl"
OUTPUT_DIR = "./exaone13b-kormed-lora"
REPO_ID = "HongKi08/HAI_Project"

# 데이터셋 로딩 및 포맷
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_prompt(example):
    return {
        "prompt": example["instruction"],
        "output": example["output"]
    }

formatted_data = dataset.map(format_prompt)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로딩 with flash_attention_2
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# k-bit 훈련 준비
model = prepare_model_for_kbit_training(model)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 토크나이즈
def tokenize(example):
    return tokenizer(
        example["prompt"],
        text_target=example["output"],
        truncation=True,
        padding="max_length",
        max_length=768
    )

tokenized_dataset = formatted_data.map(tokenize, batched=False)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none"
)

# Trainer 실행
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset
)

# 체크포인트 이어서 학습
trainer.train(resume_from_checkpoint=True)

# 모델 저장
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# Hugging Face 업로드
model.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)
