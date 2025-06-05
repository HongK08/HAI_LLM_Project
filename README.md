# HAI_Project: Medical QA with EXAONE 3.5 7.8B-Instruct

본 프로젝트는 EXAONE 3.5 7.8B-Instruct 모델을 기반으로 한 한국어 의료 QA 파인튜닝 프로젝트입니다.  
질문-지시-응답 포맷의 데이터셋을 활용해 의료 분야의 정답률과 설명 능력을 향상시키는 것을 목표로 합니다.

---
✅ 모델 변경 사유

본 프로젝트는 초기에는 Upstage/SOLAR-10.7B-Instruct-v1.0 모델을 기반으로 의료 질의응답 파인튜닝을 진행하였습니다.

하지만 다음과 같은 이유로 현재 사용 중인 모델로 전환하게 되었습니다:

추론 속도 개선SOLAR-10.7B 모델은 파라미터 수가 많아 추론 시간이 다소 길었습니다.이에 따라 보다 경량화된 구조를 갖춘 모델로 교체하여 응답 속도를 대폭 향상시켰습니다.

한국어 특화 성능 확보SOLAR 모델은 다국어 처리에 강점이 있었지만,한국어에 특화된 응답 품질을 충분히 보장하지는 못했습니다.현재 사용하는 모델은 한국어에 최적화된 학습이 선행되어,별도의 한국어 SFT 없이도 높은 정답률을 보장합니다.

또한 기존의 1차 학습 데이터(S_F1)는 제외하고,의료 QA에 특화된 2차 정제 데이터셋만을 사용하여 파인튜닝을 진행하였습니다.이를 통해 모델의 일관성과 응답 신뢰도를 더욱 높일 수 있었습니다.

🧾 학습 데이터 구성

사용된 데이터셋은 총 42만 쌍의 질문-응답 샘플로 구성되어 있으며,AI Hub에서 제공한 의료 질의응답 데이터를 기반으로 아래와 같은 트리 구조를 따릅니다:

파일 구조: 02.라벨링데이터/TL/

📁 02.라벨링데이터/TL/
├── 📁 1.질문/
│   ├── 📁 감기/
│   │   ├── 📁 예방/
│   │   │   ├── q1.json
│   │   │   └── q2.json
│   │   └── 📁 증상/
│   │       └── q1.json
│   ├── 📁 고혈압/
│   │   ├── 📁 원인/
│   │   ├── 📁 진단/
│   │   └── 📁 치료/
│   └── ...
├── 📁 2.응답/
│   ├── 📁 감기/
│   │   ├── 📁 예방/
│   │   │   ├── a1.json
│   │   │   └── a2.json
│   │   └── 📁 증상/
│   │       └── a1.json
│   ├── 📁 고혈압/
│   │   ├── 📁 원인/
│   │   ├── 📁 진단/
│   │   └── 📁 치료/
│   └── ...
...
(총 약 9,538개의 폴더)

상위 폴더는 질병명 (예: 고혈압, 당뇨병, 감기 등)

하위 폴더는 세부 카테고리 (예: 예방, 원인, 증상, 진단, 치료 등)

1.질문/에는 각 항목에 해당하는 질문 JSON 파일, 2.응답/에는 해당 질문에 대한 응답 JSON 파일이 포함됩니다.

이 구조를 flatten하여 SFT에 적합한 형식으로 전처리하고,{"instruction": "...", "output": "..."} 형태로 학습에 활용하였습니다.

🧪 데이터 전처리 스크립트

preprocess_medical.py는 AI Hub TL 구조의 질문/응답 폴더를 순회하며,
각 질문(JSON)과 응답(JSON)을 매칭한 후 instruction / output 형식으로 구성하여 .jsonl 파일로 저장하는 전처리 스크립트입니다.

예: formatted_medical_dataset.jsonl
{"instruction": "고혈압 환자가 피해야 할 음식은 무엇인가요?", "output": "김치, 짠 음식 등 나트륨이 많은 음식."}

전처리된 이 파일은 S_F2.py의 학습 입력으로 사용됩니다.

⚙️ 파인튜닝 구성 스크립트

S_F2.py는 변경된 EXAONE 기반 모델에 대해 2차 정제 데이터셋을 활용한 LoRA 기반 파인튜닝을 수행하는 메인 스크립트입니다.해당 스크립트는 다음을 포함합니다:

4bit QLoRA 세팅을 포함한 모델 로드

학습 데이터 전처리

Trainer 기반 학습 루프

safetensors 형식 저장 및 Hugging Face Hub push

preprocess_medical.py는 AI Hub TL 구조의 질문/응답 폴더를 순회하며,
각 질문(JSON)과 응답(JSON)을 매칭한 후 instruction / output 형식으로 구성하여 .jsonl 파일로 저장하는 전처리 스크립트입니다.

예: formatted_medical_dataset.jsonl
{"instruction": "고혈압 환자가 피해야 할 음식은 무엇인가요?", "output": "김치, 짠 음식 등 나트륨이 많은 음식."}

전처리된 이 파일은 S_F2.py의 학습 입력으로 사용됩니다.




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
