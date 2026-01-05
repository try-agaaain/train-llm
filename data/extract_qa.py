import json
from pathlib import Path

# 读取jsonl文件并提取所有entries中的qa_pairs
pwd = Path(__file__).parent
input_path = pwd / "train_dataset.jsonl"
output_path = pwd.parent / "artifacts" / "dataset" / "qa_dataset.json"
qa_pairs = []

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    entries = data.get('entries', [])
    for entry in entries:
        for qa in entry.get('qa_pairs', []):
            question = qa.get('question')
            answer = qa.get('content')
            if question and answer:
                qa_pairs.append({'question': question, 'answer': answer})

# 确保输出目录存在
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"已提取{len(qa_pairs)}条问答对，保存至{output_path}")
