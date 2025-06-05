# HAI_Project: Medical QA with EXAONE 3.5 7.8B-Instruct

본 프로젝트는 EXAONE 3.5 7.8B-Instruct 모델을 기반으로 한 한국어 의료 QA 파인튜닝 프로젝트입니다.  
질문-지시-응답 포맷의 데이터셋을 활용해 의료 분야의 정답률과 설명 능력을 향상시키는 것을 목표로 합니다.

---

## 📌 모델 정보

- **Base Model**: [`LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
- **파인튜닝 방식**: QLoRA (4bit 양자화 + LoRA)
- **데이터 형식**: `{"instruction": "...", "output": "..."}` (JSONL)
- **사용 분야**: 한국어 의료 질의응답, 설명형 QA, 단답형 선택 문제

---

## 📂 학습 데이터

- 사용 데이터셋: `formatted_medical_dataset.jsonl`
- 주요 구성:  
  - `instruction`: 질문 및 지시 문장  
  - `output`: 정답 또는 의료적 설명 응답

---

## ⚙️ 학습 세부 설정

- `per_device_train_batch_size`: 2  
- `gradient_accumulation_steps`: 4  
- `num_train_epochs`: 3  
- `bnb_4bit`: NF4, double quant  
- `LoRA target modules`: `["q_proj", "k_proj", "v_proj", "o_proj"]`

---

## 🧠 사용 예시

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HongKi08/HAI_Project")
tokenizer = AutoTokenizer.from_pretrained("HongKi08/HAI_Project")

prompt = "고혈압 환자가 피해야 할 음식은 무엇인가요?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
