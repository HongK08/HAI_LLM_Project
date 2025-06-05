import os, json, glob
from tqdm import tqdm

question_root = "/home/user/바탕화면/Medical/Data/Training/02.라벨링데이터/TL/1.질문/"
answer_root = "/home/user/바탕화면/Medical/Data/Training/02.라벨링데이터/TL/2.답변/"
output_path = "/home/user/바탕화면/Medical_CONF/formatted_medical_dataset.jsonl"
results = []

question_files = glob.glob(os.path.join(question_root, "**", "*.json"), recursive=True)

for q_file in tqdm(question_files):
    try:
        with open(q_file, encoding="utf-8") as f:
            qdata = json.load(f)

        rel_path = os.path.relpath(q_file, question_root)
        parts = rel_path.split(os.sep)
        if len(parts) < 4: continue

        category, name, intention = parts[0], parts[1], parts[2]
        answer_dir = os.path.join(answer_root, category, name, intention)
        if not os.path.exists(answer_dir): continue

        answer_files = glob.glob(os.path.join(answer_dir, "*.json"))
        if not answer_files: continue

        instruction = qdata.get("question", "").strip()
        if not instruction: continue

        with open(answer_files[0], encoding="utf-8") as af:
            adata = json.load(af)
        a = adata.get("answer", {})
        output = " ".join([a.get("intro", ""), a.get("body", ""), a.get("conclusion", "")]).strip()

        results.append({
            "input": "",
            "instruction": instruction,
            "output": output
        })

    except Exception as e:
        continue


with open(output_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"전처리 완료! 총 {len(results)}개 저장함")
