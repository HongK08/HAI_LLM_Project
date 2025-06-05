# 🩺 HAI_Project: Medical QA with EXAONE-3.5-7.8B-Instruct

본 프로젝트는 한국어 의료 질의답변 태스크를 위한 파인튜닝 프로젝트입니다.  
LGAI-EXAONE의 공개 모델을 기반으로, 한국어 의료 QA 성능을 극대화하는 데 목적이 있습니다.

---

## ✅ 개요

- **모델**: [EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
- **데이터**: AI Hub TL 의료 QA 데이터
- **파인튜닝 방식**: LoRA (QLoRA, 4bit) 기반 SFT
- **사용 목적**: 질병, 진단, 증상, 치료 등 의료 정보 자동 답변 시스템 구축

---

## 📂 파일 구조

```
HAI_Project/
├── README.md                   # 프로젝트 설명서
├── S_F2.py                     # 파인튜닝 메인 스크립트
└── preprocess_medical.py       # TL 구조 전처리 스크립트
```

---

## 🧾 주요 스크립트 설명

### 🔹 `preprocess_medical.py`

- `02.라벨링데이터/TL/` 내 질문/답변 JSON 파일 매칭
- `{ "instruction": 질문, "output": 답변 }` 형태의 `.jsonl` 생성

```json
{"instruction": "당뇨병은 어떤 질병인가요?", "output": "당뇨병은 혈당이 비정상적으로 높은 상태로 ..."}
```

### 🔹 `S_F2.py`

- EXAONE 모델에 대해 QLoRA 학습 수행
- 학습 후 Hugging Face 업로드 가능

---

## 🗂️ AI Hub 데이터 트리 구조 (원천 데이터 설명)

본 프로젝트에서 사용된 의료 QA 데이터는 AI Hub의 TL 구조를 기반으로 하며, 아래와 같은 디렉토리 구조를 가집니다:

```
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
├── 📁 2.답변/
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
```

- `1.질문/`: 질병명 및 세부 카테고리별 질문 파일 존재
- `2.답변/`: 동일한 구조의 답변 파일 존재
- 총 약 9,538개 폴더로 구성

> 이 구조를 flatten 하여 `{ "instruction": 질문, "output": 답변 }` 포맷 `.jsonl`로 파인튜닝에 활용

---

## ✨ 샘플 프롬프트 (사용 예시)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HongKi08/HAI_Project")
tokenizer = AutoTokenizer.from_pretrained("HongKi08/HAI_Project")

prompt = "44세 여성이 혈당의 상승 하강 폭이 불규칙하며 신체 말단에 감각이 소실되고 돌아오는 현상이 있습니다. 이는 어떤 질병입니까?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.0,
    do_sample=False
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 💬 대화형 질의 예시 (Chatbot 스타일)

#### 예시 1: 소화기계 질환

```
👤 사용자: 배가 아파요.
🤖 모델: 복통은 어떤 부위에 발생하나요? (예: 상복부, 하복부 등)
👤 사용자: 오른쪽 하복부요.
🤖 모델: 동반 증상은 있나요? (예: 발열, 메스꺼움 등)
👤 사용자: 열도 있어요.
🤖 모델: 주어진 증상은 충수염입니다. 빠른 수술적 처치를 위해 병원 진료가 필요합니다.
```

#### 예시 2: 신경계 이상

```
👤 사용자: 손끝이 저려요.
🤖 모델: 양손 모두 저리신가요, 특정 부위만 그러신가요?
👤 사용자: 왼손만요.
🤖 모델: 경추 디스크 압박 또는 말초신경 손상입니다. 정형외과 또는 신경과 진료가 필요합니다.
```

#### 예시 3: 대사성 질환

```
👤 사용자: 요즘 계속 피곤하고 몸이 무거워요.
🤖 모델: 최근 식욕 변화나 체중 변화가 있었나요?
👤 사용자: 식욕은 줄고 체중은 빠졌어요.
🤖 모델: 갑상선기능저하증입니다. 내분비내과 진료 및 혈액검사가 필요합니다.
```

---

## 🔧 시스템 프롬프트 기반 구성

```python
system_prompt = (
    "당신은 친절하고 신뢰할 수 있는 한국어 의료 상담 챗봇입니다."
    " 사용자의 질문이 모호하거나 부족한 경우, 필요한 정보를 다시 질문하여 정확한 진단과 구체적인 처방까지 유도하세요."
    " 답변은 간결하고 이해하기 쉽게 제공하세요."
)

chat_prompt = f"""<|system|>
{system_prompt}
<|user|>
복통이 있어요.
<|assistant|>"""

inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.0,
    do_sample=False
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---
## 📁 Hugging Face 저장소

- 모델 업로드 위치: [HongKi08/HAI_Project](https://huggingface.co/HongKi08/HAI_Project)

---

## 📋 작성자

- (HongKi08)
- 소속: HAI
- 지도교수 : 원유석
