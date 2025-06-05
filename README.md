# 🩺 HAI\_Project: Medical QA with EXAONE-3.5-7.8B-Instruct

본 프로젝트는 한국어 의료 질의답변 태스크를 위한 파인튜닝 프로젝트입니다. LGAI-EXAONE의 공개 모델을 기반으로, 한국어 의료 QA 성능을 극대화하는 데 목적이 있습니다.

---

## ✅ 개요

* **모델**: [EXAONE-3.5-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
* **데이터**: AI Hub 의 TL 의료 QA 데이터
* **파인튜닝 방식**: LoRA (QLoRA, 4bit) 기반 SFT
* **사용 목적**: 질병, 진단, 증상, 치료 등 의료 정보 자동 답변 시스템 개발

---

## 📂 파일 구조

```
HAI_Project/
├── README.md                   # 프로젝트 설명서
├── S_F2.py                     # 파인튜닝 메인 스크립트
├── preprocess_medical.py      # TL 구조 전처리 스크립트
├── requirements.txt           # 패키지 리스트
├── Dockerfile                 # 모델 실행용 이미지 빌드
└── docs/
    ├── server_setup.md        # 서버 구축 (CUDA + Docker)
    ├── model_merge.md         # 병합, 추론 코드 문서
    └── data_structure.md      # TL QA 트리 구조 설명
```

---

## 🧾 주요 스크립트 설명

## 🗂️ AI Hub TL 트리 구조 (원천 데이터 설명)

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

* `1.질문/`: 질병명 및 세부 카테고리별로 질문 파일(qN.json) 존재
* `2.답변/`: 동일한 구조로 질문에 대응되는 답변 파일(aN.json) 존재
* 총 약 9,538개 폴더로 구성되며, 모든 질문과 답변은 매칭 처리됨

이 구조를 flatten하여 `{ "instruction": 질문, "output": 답변 }` 형식의 `.jsonl`로 변환하여 파인튜닝에 활용하였습니다.

### 🔹 `preprocess_medical.py`

* TL 기반 `02.라벨링데이터/TL/` 폴더 내부 `1.질문/`, `2.답변/` JSON 파일을 매칭
* `{ "instruction": 질문, "output": 답변 }` 포맷으로 `.jsonl` 파일 생성
* 예시:

```json
{"instruction": "당뇨병은 어떤 질병인가요?", "output": "당뇨병은 혈당이 비정상적으로 높은 상태로 ..."}
```

### 🔹 `S_F2.py`

* EXAONE 기반 모델에 대해 LoRA 학습 진행
* `formatted_medical_dataset.jsonl` 로드
* 4bit QLoRA + LoRA 병렬 구성
* 학습 후 Hugging Face Hub push 가능

---

## 🚀 실행 순서 요약

1. AI Hub 데이터 → `preprocess_medical.py` 실행하여 `.jsonl` 생성
2. `S_F2.py` 실행하여 학습 진행 (QLoRA + LoRA)
3. 병합 후 모델 저장 → Hugging Face 업로드 가능

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
    temperature=0.0,  # 무작위성 제거
    do_sample=False   # 정확한 응답 유도
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 💬 대화형 질의 예시 (Chatbot 스타일)

LLM이 사용자의 질문이 불완전하거나 애매할 경우, 필요한 정보를 재질문한 뒤, **구체적인 진단명과 처방을 제공하는 방식**으로 응답합니다. 아래는 증상 별 질의 예시입니다:

#### 예시 1: 소화기계 질환

```
👤 사용자: 배가 아파요.
🤖 모델: 복통은 어떤 부위에 발생하나요? (예: 상복부, 하복부 등)
👤 사용자: 오른쪽 하복부요.
🤖 모델: 동반 증상은 있나요? (예: 발열, 메스꺼움 등)
👤 사용자: 열도 있어요.
🤖 모델: 주어진 증상은 충수염(맹장염)입니다. 빠른 수술적 처치를 위해 병원 진료가 필요합니다.
```

#### 예시 2: 신경계 이상

```
👤 사용자: 손끝이 저려요.
🤖 모델: 양손 모두 저리신가요, 특정 부위만 그러신가요?
👤 사용자: 왼손만요.
🤖 모델: 경추 디스크 압박 혹은 말초신경 이상이 의심됩니다. 정형외과 또는 신경과 진료가 필요합니다.
```

#### 예시 3: 대사성 질환

```
👤 사용자: 요즘 계속 피곤하고 몸이 무거워요.
🤖 모델: 최근 식욕 변화나 체중 변화가 있었나요?
👤 사용자: 식욕은 줄고 체중은 빠졌어요.
🤖 모델: 갑상선기능저하증이 의심됩니다. 내분비내과 진료 및 혈액검사가 필요합니다.
```

정확한 진단과 적절한 처방을 위해서는 반드시 의사의 진료가 필요합니다.

````
이러한 대화형 설계를 위해 시스템 프롬프트 혹은 프론트엔드 상에서 instruction 템플릿을 지정해 사용할 수 있습니다.

예시 코드 (시스템 프롬프트 방식):
```python
system_prompt = (
    "당신은 친절하고 신뢰할 수 있는 한국어 의료 상담 챗봇입니다."
    " 사용자의 질문이 모호하거나 부족한 경우, 필요한 정보를 다시 질문하여 정확한 진단과 구체적인 처방까지 유도하세요."
    " 답변은 간결하고 이해하기 쉽게 제공하세요."
)

chat_format_prompt = f"""<|system|>
{system_prompt}
<|user|>
복통이 있어요.
<|assistant|>"""

inputs = tokenizer(chat_format_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.0,  # 무작위성 제거
    do_sample=False   # 정확한 응답 유도
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
````

이와 같은 시스템 프롬프트 기반 구성은 `merge_and_unload()` 후 저장된 모델을 불러올 때도 동일하게 활용 가능합니다.

### 🔧 병합된 모델 적용 시 시스템 프롬프트 삽입

모델 병합 이후에도 아래와 같이 시스템 프롬프트를 포함하여 inference에 활용할 수 있습니다:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model_path", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./merged_model_path")

system_prompt = (
    "당신은 친절하고 신뢰할 수 있는 한국어 의료 상담 챗봇입니다."
    " 사용자의 질문이 모호하거나 부족한 경우, 필요한 정보를 다시 질문하여 정확한 진단과 적절한 의료 조치를 안내하도록 하세요."
    " 답변은 간결하고 이해하기 쉽게 제공하세요."
)

user_query = "44세 여성이 혈당의 상승 하강 폭이 불규칙하며 신체 말단에 감각이 소실되고 돌아오는 현상이 있습니다. 이는 어떤 질병입니까?"

prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_query}
<|assistant|>"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.0,  # 무작위성 제거
    do_sample=False   # 정확한 응답 유도
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

이 구조는 **모델이 유저의 모호한 요청을 인식하고, 반응을 유도하며, 명확한 진단과 처방을 제공하는 대화형 QA 시스템** 설계에 매우 유용합니다.

---

## 📦 종속 패키지

아래 모든 패키지는 `requirements.txt` 파일에 포함되어 있어, 다음 명령어로 일괄 설치할 수 있습니다:

```bash
pip install -r requirements.txt
```

### `requirements.txt` 내용 예시:

```
transformers
peft
datasets
accelerate
bitsandbytes
torch==2.1.2
sentencepiece
protobuf
huggingface_hub
safetensors
llama-cpp-python
einops
scipy
ninja
tqdm
loguru
```

> `llama-cpp-python`, `sentencepiece`, `protobuf` 등은 GGUF 변환 및 추론 시 사용됩니다.

### 🔗 종속 라이브러리 공식 설치 가이드

* [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
* [transformers 문서](https://huggingface.co/docs/transformers/index)
* [bitsandbytes 설치 가이드](https://github.com/TimDettmers/bitsandbytes)
* [llama.cpp 설치법](https://github.com/ggerganov/llama.cpp)
* [sentencepiece 설치법](https://github.com/google/sentencepiece)
* [safetensors 개요](https://huggingface.co/docs/safetensors/index)

환경에 따라 C++ 빌드 도구(ninja, cmake 등), `cuda` 툴킷, `g++` 등의 사전 설치가 요구될 수 있습니다.

---

## 📁 Hugging Face 저장소

* 모델 업로드 위치: [HongKi08/HAI\_Project](https://huggingface.co/HongKi08/HAI_Project)

---

## 📋 작성자

* 프로젝트 총괄: PM & LLM 개발 책임자 (HongKi08)
* 소속: 대학원 연구실 기반 HAI 프로젝트 팀
