
# 🩺 HAI_Project: Medical QA with EXAONE-3.5-7.8B-Instruct

본 프로젝트는 한국어 의료 질의응답 태스크를 위한 파인튜닝 프로젝트입니다.  
LGAI-EXAONE의 공개 모델을 기반으로, 한국어 의료 QA 성능을 극대화하는 데 목적이 있습니다.

---

## ✅ 개요

- **모델**: [EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)  
- **데이터**: AI Hub TL 의료 QA 데이터  
- **파인튜닝 방식**: LoRA (QLoRA, 4bit) 기반 SFT  
- **사용 목적**: 질병, 진단, 증상, 치료 등 의료 정보 자동 답변 시스템 개발

---

## 📂 파일 구조

```
HAI_Project/
├── README.md
├── S_F2.py
├── preprocess_medical.py
└── docs/
    ├── model_merge.md
    └── data_structure.md
```

---

## 🧾 주요 스크립트 설명

### 🔹 `preprocess_medical.py`

- TL 구조의 질문/답변 JSON 파일을 매칭  
- `{ "instruction": 질문, "output": 답변 }` 형식의 `.jsonl` 파일 생성

```json
{"instruction": "당뇨병은 어떤 질병인가요?", "output": "당뇨병은 혈당이 비정상적으로 높은 상태로 ..."}
```

### 🔹 `S_F2.py`

- EXAONE 기반 모델에 대해 LoRA 학습 진행  
- `formatted_medical_dataset.jsonl` 로드  
- 4bit QLoRA + LoRA 병렬 구성

---

## 🗂️ AI Hub TL 트리 구조

```
📁 02.라벨링데이터/TL/
├── 1.질문/감기/예방/q1.json
├── 2.답변/감기/예방/a1.json
...
```

- 동일한 트리 구조로 질문-답변 매칭  
- 총 약 9,538개 폴더  
- 최종 출력은 `{ "instruction": ..., "output": ... }` 포맷

---

## 🚀 실행 순서 요약

1. `preprocess_medical.py` 실행 → `.jsonl` 생성  
2. `S_F2.py` 실행 → QLoRA + LoRA 학습  
3. 모델 병합 및 Hugging Face 업로드

---

## ✨ 샘플 프롬프트

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HongKi08/HAI_Project")
tokenizer = AutoTokenizer.from_pretrained("HongKi08/HAI_Project")

prompt = "44세 여성이 혈당의 상승 하강 폭이 불규칙하며 신체 말단에 감각이 소실되고 돌아오는 현상이 있습니다. 이는 어떤 질병입니까?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 💬 대화형 질의 예시

#### 예시 1: 소화기계 질환

👤 배가 아파요.  
🤖 복통은 어떤 부위에 발생하나요?  
👤 오른쪽 하복부요.  
🤖 발열, 메스꺼움 같은 증상은 있나요?  
👤 열도 있어요.  
🤖 충수염(맹장염)입니다. 빠른 수술적 처치를 위해 병원 진료가 필요합니다.

#### 예시 2: 신경계 이상

👤 손끝이 저려요.  
🤖 양손 모두 저리신가요, 특정 부위만 그런가요?  
👤 왼손만이요.  
🤖 경추 디스크 압박 또는 말초신경 이상입니다. 정형외과 또는 신경과 진료가 필요합니다.

#### 예시 3: 대사성 질환

👤 계속 피곤하고 몸이 무거워요.  
🤖 식욕 변화나 체중 변화가 있었나요?  
👤 식욕은 줄고 체중은 빠졌어요.  
🤖 갑상선기능저하증입니다. 내분비내과 진료 및 혈액검사가 필요합니다.

> ※ 정확한 진단과 처방을 위해 실제 의사의 진료가 반드시 필요합니다.

---

## 🧠 시스템 프롬프트 예시

```python
system_prompt = (
    "당신은 친절하고 신뢰할 수 있는 한국어 의료 상담 챗봇입니다. "
    "질문이 모호할 경우 추가 질문을 하여 진단과 처방을 구체화하세요."
)

chat_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n복통이 있어요.\n<|assistant|>"

inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📦 주요 패키지

- `transformers`  
- `peft`  
- `datasets`  
- `accelerate`  
- `bitsandbytes`  
- `torch==2.7.0`  ← CUDA 12.8 지원 최신 버전  
- `sentencepiece`  
- `protobuf`  
- `huggingface_hub`  
- `safetensors`  
- `einops`  
- `scipy`  
- `ninja`  
- `tqdm`  
- `loguru`

> 일부 패키지는 GGUF 추론 또는 양자화 작업에 필요합니다.

---

## 📁 Hugging Face 저장소

- [HongKi08/HAI_Project](https://huggingface.co/HongKi08/HAI_Project)

---

## 📋 작성자

- PM & LLM 개발 책임자: **HongKi08**  
- 소속: 대학원 연구실 기반 HAI 프로젝트 팀
